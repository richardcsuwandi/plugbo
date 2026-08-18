"""AlphaBO plugin registry: CAKE, TuRBO, πBO, LLAMBO, …"""

from .base import (
    SLOT_PRIOR,
    SLOT_REGION,
    SLOT_SAMPLER,
    SLOT_SURROGATE,
    LenzPlugin,
    PluginError,
)
from .registry import (
    all_plugins,
    enabled_plugins,
    occupant,
    plugin_by_name,
    plugins_for_slot,
    surrogate_names,
)

__all__ = [
    "SLOT_PRIOR",
    "SLOT_REGION",
    "SLOT_SAMPLER",
    "SLOT_SURROGATE",
    "LenzPlugin",
    "PluginError",
    "all_plugins",
    "enabled_plugins",
    "occupant",
    "plugin_by_name",
    "plugins_for_slot",
    "surrogate_names",
]
