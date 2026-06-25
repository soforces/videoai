"""Imports every plugin module so its @registry.register decorator runs.

Call load_plugins() once at process startup (app.py and worker.py both do this)
before the pipeline/decision engine needs to look anything up in the registry.
New plugins are picked up automatically just by adding their import here -
nothing else in core/ needs to change.
"""
from __future__ import annotations

_loaded = False


def load_plugins() -> None:
    global _loaded
    if _loaded:
        return

    import plugins.export.exporters  # noqa: F401
    import plugins.upscale.realesrgan_plugin  # noqa: F401
    import plugins.upscale.swinir_plugin  # noqa: F401
    import plugins.faces.gfpgan_plugin  # noqa: F401
    import plugins.faces.codeformer_plugin  # noqa: F401
    import plugins.restoration.nafnet_plugin  # noqa: F401
    import plugins.restoration.restormer_plugin  # noqa: F401
    import plugins.background.rembg_plugin  # noqa: F401
    import plugins.segmentation.sam2_plugin  # noqa: F401
    import plugins.ocr.paddleocr_plugin  # noqa: F401
    import plugins.interpolation.rife_plugin  # noqa: F401
    import plugins.analysis_models.musiq_plugin  # noqa: F401

    _loaded = True
