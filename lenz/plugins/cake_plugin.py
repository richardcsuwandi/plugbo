"""CAKE as a surrogate-slot plugin. Algorithm stays in ``lenz/cake.py``."""

from __future__ import annotations

from pathlib import Path

from .. import cake
from ..llm_config import add_llm_flags, resolved_plugin_llm, spec_complete
from ..space import Encoder
from ..state import Frame, Trial
from .base import SLOT_SURROGATE, LenzPlugin, PluginError


class CakePlugin(LenzPlugin):
    name = "cake"
    slot = SLOT_SURROGATE

    def default_state(self) -> dict:
        return cake.default_state()

    def add_parser(self, sub, state_parent) -> None:
        evk = sub.add_parser("evolve-kernels", parents=[state_parent])
        evk.add_argument("--force", action="store_true")
        sub.add_parser("kernel-population", parents=[state_parent])

    def add_create_args(self, parser) -> None:
        add_llm_flags(parser, flag_prefix="--kernel-llm", dest_prefix="kernel_llm", help_suffix="optional CAKE override; default is Sara / --llm-*")
        parser.add_argument("--kernel-population-size", type=int, default=None)
        parser.add_argument("--kernel-init-after", type=int, default=None)
        parser.add_argument("--kernel-evolve-every", type=int, default=None)
        parser.add_argument("--kernel-freeze-fraction", type=float, default=None)
        parser.add_argument("--kernel-num-crossover", type=int, default=None)
        parser.add_argument("--kernel-mutation-prob", type=float, default=None)

    def apply_create_args(self, frame: Frame, args) -> None:
        cake.apply_kernel_args(frame, args)

    def commands(self):
        return {
            "evolve-kernels": self._evolve,
            "kernel-population": self._population,
        }

    def on_observe(self, frame: Frame, trial: Trial) -> None:
        cake.maybe_evolve(frame, Encoder(frame.space))

    def summary(self, frame: Frame) -> dict:
        blob = cake.state(frame)
        targets = cake.cake_targets(frame)
        best = cake.get_best_kernel(frame)
        override = blob.get("kernel_llm") or {}
        resolved = resolved_plugin_llm(frame, override)
        out = {
            "kernel_targets": targets,
            "best_kernels": best,
            "kernel_population_size": sum(len(blob["kernel_populations"].get(t, [])) for t in targets),
            "kernel_llm": {k: v for k, v in resolved.items() if k != "api_key"} if spec_complete(resolved) else {},
            "kernel_llm_source": (
                "override" if spec_complete(override) else ("default" if spec_complete(frame.default_llm) else "unset")
            ),
        }
        if frame.shelf.objectives:
            primary = frame.shelf.objectives[0].metric
            evo = blob["kernel_evolution_states"].get(primary, {})
            out["kernel_generation"] = evo.get("generation", 0)
            out["kernel_frozen"] = evo.get("frozen", False)
            out["best_kernel"] = best.get(primary) if isinstance(best, dict) else best
        return out

    def prompt_path(self) -> Path | None:
        path = Path(__file__).parent / "cake.md"
        return path if path.exists() else None

    def _evolve(self, frame: Frame, args):
        if frame.shelf.surrogate != "cake":
            raise PluginError("evolve-kernels requires surrogate 'cake'; run 'set-surrogate --surrogate cake' first")
        ran = cake.maybe_evolve(frame, Encoder(frame.space), force=bool(args.force))
        return frame, {"evolved": ran, "surrogate": "cake", **self.summary(frame)}

    def _population(self, frame: Frame, args):
        if frame.shelf.surrogate != "cake":
            raise PluginError("kernel-population requires surrogate 'cake'")
        blob = cake.state(frame)
        targets = cake.cake_targets(frame)
        return None, {
            "targets": targets,
            "populations": {t: blob["kernel_populations"].get(t, []) for t in targets},
            "best": cake.get_best_kernel(frame),
            "evolution_states": blob["kernel_evolution_states"],
        }
