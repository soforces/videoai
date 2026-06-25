"""Plugin architecture: every AI capability (upscale, face restore, restoration,
segmentation, background removal, OCR, interpolation, export) is implemented as a
plugin subclassing BasePlugin and registered under a category in PluginRegistry.

New models are added by dropping a new module under plugins/<category>/ and
registering it in plugins/registry.py - the core pipeline never needs to change.
"""
from plugins.base import BasePlugin, PluginNotInstalledError
from plugins.registry import PluginRegistry, registry

__all__ = ["BasePlugin", "PluginNotInstalledError", "PluginRegistry", "registry"]
