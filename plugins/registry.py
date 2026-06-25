from __future__ import annotations

from plugins.base import BasePlugin


class PluginRegistry:
    """Category -> name -> plugin instance. Plugins self-register via @registry.register."""

    def __init__(self):
        self._plugins: dict[str, dict[str, BasePlugin]] = {}

    def register(self, plugin_cls: type[BasePlugin]):
        instance = plugin_cls()
        self._plugins.setdefault(instance.category, {})[instance.name] = instance
        return plugin_cls

    def get(self, category: str, name: str) -> BasePlugin:
        try:
            return self._plugins[category][name]
        except KeyError as exc:
            raise KeyError(f"No plugin registered for category={category!r} name={name!r}") from exc

    def list_category(self, category: str) -> dict[str, BasePlugin]:
        return dict(self._plugins.get(category, {}))

    def list_all(self) -> dict[str, dict[str, BasePlugin]]:
        return {cat: dict(plugins) for cat, plugins in self._plugins.items()}


registry = PluginRegistry()
