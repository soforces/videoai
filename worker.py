#!/usr/bin/env python3
"""Huey consumer process. Run this alongside app.py:

  python worker.py

This is the process that actually loads AI models and executes pipeline jobs;
keep it running so models stay persistently loaded in VRAM across jobs instead
of being reloaded per request.

Structured as a tiny self-restarting supervisor: `python worker.py` (no args)
takes the single-instance lock and spawns the real consumer as a child process
(`python worker.py --consumer`). A CUDA OOM can leave the GPU's CUDA context
permanently unusable for the rest of that process's life on this driver/torch
combo (a known PyTorch/CUDA limitation, not something fixable by freeing more
memory in-process - see core/tasks.py's `_is_cuda_oom` handling). When that
happens, the consumer child deliberately exits non-zero, and the supervisor
restarts it as a fresh process with a clean GPU context for the next job,
instead of every subsequent job failing identically forever.
"""
import os

# Must be set before torch's CUDA allocator is initialized (any of the model
# plugin modules importing torch at module level would trigger that).
# `expandable_segments:True` (PyTorch's own suggestion in the CUDA OOM error
# message) relies on virtual-memory APIs that are unreliable on Windows in
# this torch/driver combo and was observed making the allocator's own
# reported "allocated" byte count exceed the physical 4GB card - garbage
# collection + a split-size cap is the conservative, Windows-safe equivalent.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6,max_split_size_mb:128")

import subprocess
import sys

from core.singleton_lock import SingleInstanceError, acquire_singleton_lock


def _run_consumer() -> None:
    from huey.consumer import Consumer

    from core.queue_app import huey
    from core import tasks  # noqa: F401 - registers @huey.task() functions
    from plugins.load_all import load_plugins

    load_plugins()
    consumer = Consumer(huey, workers=1, worker_type="thread")
    consumer.run()


def _supervise() -> None:
    try:
        acquire_singleton_lock("worker")
    except SingleInstanceError as exc:
        print(f"[worker.py] {exc}", file=sys.stderr)
        sys.exit(1)

    while True:
        proc = subprocess.Popen([sys.executable, __file__, "--consumer"])
        returncode = proc.wait()
        if returncode == 0:
            break
        print(
            f"[worker.py] consumer process exited with code {returncode} "
            f"(a CUDA OOM likely left the GPU context unusable) - restarting "
            f"with a fresh process for the next job...",
            file=sys.stderr,
        )


if __name__ == "__main__":
    if "--consumer" in sys.argv:
        _run_consumer()
    else:
        _supervise()
