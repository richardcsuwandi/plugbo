"""LLAMBO-style warmstart and candidate sampling (Liu et al., ICLR 2024).

Does not replace the GP. Occupies the sampler slot when enabled; Sara can
also call ``lenz llambo sample`` / ``warmstart`` and then ``lenz score``.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm.base import Message

from ..llm_config import add_llm_flags, client_from_spec, resolved_plugin_llm, spec_complete, spec_from_args
from ..space import Encoder
from ..state import Frame
from .base import SLOT_SAMPLER, LenzPlugin, PluginError

SAMPLER_LLM_TIMEOUT = 90.0


class LlamboPlugin(LenzPlugin):
    name = "llambo"
    slot = SLOT_SAMPLER

    def default_state(self) -> dict:
        return {
            "llm": {},
            "context": "",
            "last_candidates": [],
        }

    def add_parser(self, sub, state_parent) -> None:
        p = sub.add_parser("llambo", parents=[state_parent])
        inner = p.add_subparsers(dest="llambo_cmd", required=True)
        ws = inner.add_parser("warmstart")
        ws.add_argument("--n", type=int, default=5)
        ws.add_argument("--context", default=None)
        sm = inner.add_parser("sample")
        sm.add_argument("--n", type=int, default=8)
        sm.add_argument("--context", default=None)
        cfg = inner.add_parser("set-llm")
        cfg.add_argument("--provider", dest="llambo_llm_provider", required=True)
        cfg.add_argument("--model", dest="llambo_llm_model", required=True)
        cfg.add_argument("--base-url", dest="llambo_llm_base_url", default=None)
        cfg.add_argument("--api-key-env", dest="llambo_llm_api_key_env", default=None)
        cfg.add_argument("--extra-body", dest="llambo_llm_extra_body", default=None)
        inner.add_parser("status")

    def add_create_args(self, parser) -> None:
        add_llm_flags(
            parser,
            flag_prefix="--sampler-llm",
            dest_prefix="sampler_llm",
            help_suffix="optional LLAMBO override; default is Sara / --llm-*",
        )

    def apply_create_args(self, frame: Frame, args) -> None:
        spec = spec_from_args(args, "sampler_llm")
        if not spec_complete(spec):
            return
        self.blob(frame)["llm"] = spec

    def commands(self):
        return {"llambo": self._dispatch}

    def propose(self, frame: Frame, encoder: Encoder, q: int, bounds: torch.Tensor) -> list[dict] | None:
        if frame.shelf.sampler != "llambo":
            return None
        return self.sample(frame, encoder, n=q)

    def summary(self, frame: Frame) -> dict:
        blob = self.blob(frame)
        override = blob.get("llm") or {}
        resolved = resolved_plugin_llm(frame, override)
        return {
            "llm": {k: v for k, v in resolved.items() if k != "api_key"} if spec_complete(resolved) else {},
            "llm_source": "override" if spec_complete(override) else ("default" if spec_complete(frame.default_llm) else "unset"),
            "n_last_candidates": len(blob.get("last_candidates") or []),
            "has_context": bool(blob.get("context")),
        }

    def prompt_path(self) -> Path | None:
        path = Path(__file__).parent / "llambo.md"
        return path if path.exists() else None

    def _dispatch(self, frame: Frame, args):
        encoder = Encoder(frame.space)
        cmd = args.llambo_cmd
        if cmd == "set-llm":
            spec = spec_from_args(args, "llambo_llm")
            if not spec_complete(spec):
                raise PluginError("llambo set-llm requires --provider and --model")
            self.blob(frame)["llm"] = spec
            frame.log_event("llambo", action="set-llm")
            return frame, self.summary(frame)
        if cmd == "status":
            return None, {"sampler": frame.shelf.sampler, **self.summary(frame)}
        if cmd in ("warmstart", "sample"):
            if args.context:
                self.blob(frame)["context"] = Path(args.context).read_text()
            n = int(args.n)
            packed = self.sample(frame, encoder, n=n, warmstart=(cmd == "warmstart"))
            return frame, packed
        raise PluginError(f"unknown llambo subcommand '{cmd}'")

    def sample(self, frame: Frame, encoder: Encoder, n: int, warmstart: bool = False) -> list[dict]:
        blob = self.blob(frame)
        client = self._client(frame)
        context = blob.get("context") or _read_local_context()
        prompt = _build_prompt(frame, n, context, warmstart=warmstart)
        resp = client.chat(
            [Message(role="user", content=prompt)],
            tools=[],
            system=_SYSTEM,
        )
        configs = _parse_configs(resp.content, frame)
        packed = []
        for cfg in configs[:n]:
            x = encoder.encode(cfg)
            packed.append(
                {
                    "config": cfg,
                    "x_gp": x.detach().tolist(),
                    "acquisition_values": {},
                    "trial_id": None,
                    "acqf": "llambo",
                }
            )
        blob["last_candidates"] = packed
        frame.log_event("llambo", action="sample" if not warmstart else "warmstart", n=len(packed))
        return packed

    def _client(self, frame: Frame):
        spec = resolved_plugin_llm(frame, self.blob(frame).get("llm"))
        if not spec_complete(spec):
            raise PluginError(
                "llambo requires an LLM: it defaults to Sara / `lenz set-llm`, "
                "or pass --sampler-llm-provider / --sampler-llm-model, "
                "or run 'lenz llambo set-llm --provider ... --model ...'"
            )
        return client_from_spec(spec, timeout=SAMPLER_LLM_TIMEOUT)


_SYSTEM = (
    "You propose candidate configurations for black-box optimization. "
    "Reply with a JSON array of objects whose keys are the parameter names. "
    "No markdown, no commentary."
)


def _read_local_context() -> str:
    path = Path("context.md")
    if path.exists():
        return path.read_text()
    return ""


def _build_prompt(frame: Frame, n: int, context: str, warmstart: bool) -> str:
    space = frame.space.to_json()
    rows = []
    for t in frame.observed_trials()[-40:]:
        rows.append({"config": t.config, "metrics": t.metrics})
    parts = [
        f"Search space (JSON):\n{json.dumps(space)}",
        f"Objectives: {json.dumps([{'metric': o.metric, 'minimize': o.minimize} for o in frame.shelf.objectives])}",
    ]
    if context.strip():
        parts.append(f"Problem context:\n{context.strip()}")
    if rows and not warmstart:
        parts.append(f"Observed trials (most recent last):\n{json.dumps(rows)}")
        parts.append(f"Propose {n} new configurations that are likely to improve the objective. Stay inside the bounds.")
    else:
        parts.append(
            f"No (or ignored) trial history. Propose {n} diverse, domain-typical starting configurations inside the bounds."
        )
    parts.append('Return only a JSON array, e.g. [{"x": 1.0}, {"x": 2.0}].')
    return "\n\n".join(parts)


def _parse_configs(text: str, frame: Frame) -> list[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise PluginError(f"llambo response was not a JSON array: {text[:200]!r}")
    raw = json.loads(text[start : end + 1])
    if not isinstance(raw, list):
        raise PluginError("llambo response must be a JSON array of configs")
    out = []
    for cfg in raw:
        if not isinstance(cfg, dict):
            continue
        try:
            frame.space.validate_config_keys(cfg)
        except Exception:
            continue
        out.append(cfg)
    if not out:
        raise PluginError("llambo produced no configs that match the search space")
    return out
