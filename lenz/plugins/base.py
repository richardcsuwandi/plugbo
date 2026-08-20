"""Plugin protocol for PlugBO / lenz.

A plugin occupies one slot of the live policy (surrogate, region, sampler,
prior) and stores its own blob under ``frame.plugins[name]``. Core never
imports plugin internals: it calls hooks, and plugins register CLI verbs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch

from ..space import Encoder
from ..state import Frame, Trial

SLOT_SURROGATE = "surrogate"
SLOT_REGION = "region"
SLOT_SAMPLER = "sampler"
SLOT_PRIOR = "prior"

SLOTS = (SLOT_SURROGATE, SLOT_REGION, SLOT_SAMPLER, SLOT_PRIOR)

CommandFn = Callable[[Frame, Any], tuple[Frame | None, Any]]


class PluginError(ValueError):
    pass


class LenzPlugin:
    """One drop-in method. Subclass and register in ``registry.py``."""

    name: str = ""
    slot: str = ""

    def default_state(self) -> dict:
        return {}

    def add_parser(self, sub, state_parent) -> None:
        """Register ``lenz <verb>`` subparsers. Optional."""

    def add_create_args(self, parser) -> None:
        """Extra flags on ``lenz create`` / ``set-surrogate``. Optional."""

    def apply_create_args(self, frame: Frame, args) -> None:
        """Write create/set-surrogate flags into ``frame.plugins[name]``. Optional."""

    def commands(self) -> dict[str, CommandFn]:
        """Map CLI dest ``command`` -> ``(frame, args) -> (frame|None, result)``."""
        return {}

    def on_observe(self, frame: Frame, trial: Trial) -> None:
        """Called after a successful observe / submit-with-metrics, if this
        plugin occupies its slot (or, for bookkeeping plugins, if enabled).
        """

    def active_bounds(self, frame: Frame, encoder: Encoder) -> torch.Tensor | None:
        """Region plugins: return (2, d) GP-space bounds, or None to fall back."""
        return None

    def wrap_acqf(self, acqf, frame: Frame, encoder: Encoder):
        """Prior plugins: wrap a BoTorch acquisition. Default: identity."""
        return acqf

    def propose(self, frame: Frame, encoder: Encoder, q: int, bounds: torch.Tensor) -> list[dict] | None:
        """Sampler plugins: return packed suggestions, or None to fall back."""
        return None

    def prompt_path(self) -> Path | None:
        path = Path(__file__).parent / f"{self.name}.md"
        return path if path.exists() else None

    def summary(self, frame: Frame) -> dict:
        return {}

    def blob(self, frame: Frame) -> dict:
        store = frame.plugins.setdefault(self.name, self.default_state())
        for key, value in self.default_state().items():
            if key not in store:
                store[key] = dict(value) if isinstance(value, dict) else (
                    list(value) if isinstance(value, list) else value
                )
        return store
