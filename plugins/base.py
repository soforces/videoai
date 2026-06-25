"""Base plugin interface.

A plugin loads its model exactly once (in `load`) and keeps it resident in memory
or VRAM for the lifetime of the worker process. The pipeline calls `process` per
frame (or `process_batch` for batched inference); it must NEVER reload the model
per frame/job - that is the #1 rule for hitting the <60s performance target.

This machine's GPU has only 4GB VRAM, far less than the combined peak footprint
of every plugin this project registers (Real-ESRGAN, SwinIR, GFPGAN, CodeFormer,
NAFNet, Restormer, SAM2, MUSIQ, CLIP...). "Never reload per frame" is still
honored *within a job*, but across jobs `ensure_loaded()` now evicts the
least-recently-used *other* GPU-resident plugin on a CUDA OOM and retries,
rather than letting every plugin a long-running worker has ever touched pile up
in VRAM forever. See `unload()`.
"""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class PluginNotInstalledError(RuntimeError):
    """Raised when a plugin's dependencies/checkpoints are not installed.

    This is intentionally loud and explicit rather than silently falling back to
    a mock - callers must catch it and either skip the stage or surface the error.
    """

    def __init__(self, plugin_name: str, install_hint: str):
        self.plugin_name = plugin_name
        self.install_hint = install_hint
        super().__init__(f"Plugin '{plugin_name}' is not installed/available. {install_hint}")


class BasePlugin(ABC):
    category: str = "uncategorized"
    name: str = "base"
    #: Whether this plugin has real, runnable model code (vs. an interface stub).
    implemented: bool = True

    def __init__(self):
        self._load_lock = threading.Lock()
        self._loaded = False
        self.last_used: float = 0.0

    def ensure_loaded(self) -> None:
        self.last_used = time.time()
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            self._load_with_eviction_retry()

    def _load_with_eviction_retry(self) -> None:
        run_with_gpu_oom_retry(self.load, exclude=self)
        self._loaded = True

    @abstractmethod
    def load(self) -> None:
        """Load model weights onto device. Called at most once per process
        (more precisely: at most once between an eviction and the next call)."""

    def unload(self) -> None:
        """Release this plugin's GPU memory so another plugin can load.
        Default is a no-op (CPU-only plugins, e.g. PaddleOCR, have nothing to
        free); GPU-backed plugins override this to del their model reference(s)
        and call torch.cuda.empty_cache(). Must leave the plugin in a state
        where ensure_loaded() will call load() again cleanly."""
        self._loaded = False

    @abstractmethod
    def process(self, image: Any, **kwargs) -> Any:
        """Run inference on a single in-memory image (numpy array, BGR or RGB per plugin docstring)."""

    def process_batch(self, images: list[Any], **kwargs) -> list[Any]:
        """Default batch implementation: sequential process(). Override for true batched inference."""
        self.ensure_loaded()
        return [self.process(img, **kwargs) for img in images]

    def process_file(self, src: Path, dst: Path, **kwargs) -> None:
        """Convenience wrapper for plugins driven by file paths instead of arrays."""
        raise NotImplementedError(f"{self.name} does not support process_file")


def run_with_gpu_oom_retry(fn, exclude: BasePlugin):
    """Calls `fn()` (a plugin's load() or process() call); on CUDA OOM, evicts
    the least-recently-used *other* GPU-resident plugin and retries, repeating
    until `fn()` succeeds or there is nothing left to evict (at which point the
    OutOfMemoryError propagates to the caller, which core/tasks.py reports as
    a normal job error). `exclude` is the plugin `fn` belongs to, so eviction
    never unloads the very plugin that's trying to run right now.

    This covers OOM both while *loading* a model (weights are usually small)
    and while *running inference* (where this project's heaviest single
    allocations actually happen - e.g. SwinIR's tiled forward pass) - a
    pure load()-time retry would miss the second, far more common case.
    """
    import torch

    while True:
        try:
            return fn()
        except torch.cuda.OutOfMemoryError:
            if not _evict_least_recently_used_gpu_plugin(exclude):
                raise


def _evict_least_recently_used_gpu_plugin(exclude: BasePlugin) -> bool:
    """Finds the least-recently-used loaded GPU-resident plugin (other than
    `exclude`) across the whole registry and unloads it. Returns False if
    there is nothing left to evict."""
    import torch

    from plugins.registry import registry

    candidates = [
        plugin
        for plugins_in_category in registry.list_all().values()
        for plugin in plugins_in_category.values()
        if plugin is not exclude and plugin._loaded and getattr(plugin, "device", None) == "cuda"
    ]
    if not candidates:
        return False

    victim = min(candidates, key=lambda p: p.last_used)
    victim.unload()
    torch.cuda.empty_cache()
    return True
