"""Discover installed plugins. Adding a method is: new file + one line here."""

from __future__ import annotations

from .base import SLOT_SURROGATE, LenzPlugin, PluginError
from .cake_plugin import CakePlugin
from .llambo import LlamboPlugin
from .pibo import PiboPlugin
from .turbo import TurboPlugin
from ..state import Frame

_PLUGINS: list[LenzPlugin] | None = None


def all_plugins() -> list[LenzPlugin]:
    global _PLUGINS
    if _PLUGINS is None:
        _PLUGINS = [CakePlugin(), TurboPlugin(), PiboPlugin(), LlamboPlugin()]
    return _PLUGINS


def plugin_by_name(name: str) -> LenzPlugin:
    for plugin in all_plugins():
        if plugin.name == name:
            return plugin
    raise PluginError(f"unknown plugin '{name}'")


def plugins_for_slot(slot: str) -> list[LenzPlugin]:
    return [p for p in all_plugins() if p.slot == slot]


def surrogate_names() -> list[str]:
    return ["fixed"] + [p.name for p in plugins_for_slot(SLOT_SURROGATE)]


def occupant(frame: Frame, slot: str) -> LenzPlugin | None:
    """The plugin currently filling ``slot``, or None for the core default."""
    shelf = frame.shelf
    if slot == SLOT_SURROGATE:
        name = shelf.surrogate
        if not name or name == "fixed":
            return None
    elif slot == "region":
        name = shelf.region
        if not name or name == "box":
            return None
    elif slot == "sampler":
        name = shelf.sampler
        if not name or name == "botorch":
            return None
    elif slot == "prior":
        name = shelf.prior
        if not name or name == "none":
            return None
    else:
        return None
    try:
        plugin = plugin_by_name(name)
    except PluginError:
        return None
    return plugin if plugin.slot == slot else None


def enabled_plugins(frame: Frame) -> list[LenzPlugin]:
    out = []
    seen = set()
    for slot in ("surrogate", "region", "sampler", "prior"):
        plugin = occupant(frame, slot)
        if plugin is not None and plugin.name not in seen:
            seen.add(plugin.name)
            out.append(plugin)
    return out
