"""Single-instance guard so a second `worker.py` or `app.py` can never start
while one is already running and silently double up GPU/VRAM usage or fight
over the same SQLite-backed queue/job-status files.

Uses an OS-level exclusive lock (`msvcrt.locking`) on a sentinel file under
the project root rather than a plain PID file: the lock is tied to the open
file handle, so Windows releases it automatically the instant the holding
process exits for any reason - including a hard crash (e.g. CUDA OOM) - with
no stale-lock cleanup logic needed.

The PID is recorded in a *separate*, unlocked sidecar file purely for
diagnostics (the error message below): Windows denies other handles read
access to a locked byte range too, not just write/lock attempts, so the PID
can't be read back out of the lock file itself while another process holds it.
"""
from __future__ import annotations

import atexit
import msvcrt
import os
from pathlib import Path

LOCK_DIR = Path(__file__).resolve().parent.parent


class SingleInstanceError(RuntimeError):
    pass


def acquire_singleton_lock(name: str) -> None:
    """Raises SingleInstanceError if another process already holds the lock
    for `name` (e.g. "worker" or "app"). Call once at process startup, before
    any heavy initialization (model loading, Flask binding)."""
    lock_path = LOCK_DIR / f".{name}.lock"
    pid_path = LOCK_DIR / f".{name}.pid"

    # msvcrt.locking locks the byte range starting at the file's CURRENT
    # position, so every opener must seek(0) first to contend over the same
    # byte - relying on the implicit position "a+" opens at (which varies
    # with file size) would let two processes lock disjoint, non-conflicting
    # byte ranges and both "succeed".
    if not lock_path.exists():
        lock_path.write_bytes(b"\0")
    fh = open(lock_path, "r+")
    fh.seek(0)
    try:
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        fh.close()
        existing_pid = pid_path.read_text(encoding="utf-8").strip() if pid_path.exists() else "?"
        raise SingleInstanceError(
            f"Another '{name}' process already appears to be running "
            f"(recorded PID {existing_pid}). "
            f"Stop it first (Windows: Stop-Process -Id {existing_pid} -Force) "
            f"before starting a new one."
        )
    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    atexit.register(_release, fh)


def _release(fh) -> None:
    try:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
    except OSError:
        pass
    finally:
        fh.close()
