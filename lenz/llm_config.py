"""Shared LLM spec for plugin calls (CAKE, LLAMBO, …).

Plugins that talk to a model resolve in this order:

1. A complete plugin-specific override (``--kernel-llm-*``, ``llambo set-llm``, …)
2. ``frame.default_llm``, set by ``lenz create --llm-*``, ``lenz set-llm``, or Sara
3. ``agent_llm.json`` next to ``state.json`` (written by ``sara run``)

TuRBO and πBO do not call an LLM. Incomplete specs are left incomplete; the
caller decides whether to skip (CAKE) or error (LLAMBO). Never stores API keys.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from llm.factory import get_client

from .state import Frame

AGENT_LLM_FILENAME = "agent_llm.json"

SPEC_KEYS = ("provider", "model", "base_url", "api_key_env", "extra_body")

_PROVIDER_API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openai-compatible": "OPENAI_API_KEY",
    "ollama": None,
}


def default_api_key_env(provider: str | None) -> str | None:
    return _PROVIDER_API_KEY_ENV.get(provider or "")


def spec_complete(spec: dict | None) -> bool:
    return bool(spec and spec.get("provider") and spec.get("model"))


def normalize_spec(raw: dict | None) -> dict:
    if not raw:
        return {}
    extra = raw.get("extra_body")
    if isinstance(extra, str) and extra.strip():
        extra = json.loads(extra)
    elif extra == "" or extra == {}:
        extra = None
    out = {
        "provider": raw.get("provider"),
        "model": raw.get("model"),
        "base_url": raw.get("base_url"),
        "api_key_env": raw.get("api_key_env"),
        "extra_body": extra,
    }
    if not out["api_key_env"]:
        out["api_key_env"] = default_api_key_env(out["provider"])
    return out


def spec_from_args(args: Any, dest_prefix: str) -> dict:
    """Read ``{dest_prefix}_provider`` / ``_model`` / … off an argparse namespace."""
    provider = getattr(args, f"{dest_prefix}_provider", None)
    model = getattr(args, f"{dest_prefix}_model", None)
    if not provider or not model:
        return {}
    return normalize_spec(
        {
            "provider": provider,
            "model": model,
            "base_url": getattr(args, f"{dest_prefix}_base_url", None),
            "api_key_env": getattr(args, f"{dest_prefix}_api_key_env", None),
            "extra_body": getattr(args, f"{dest_prefix}_extra_body", None),
        }
    )


def add_llm_flags(parser, *, flag_prefix: str, dest_prefix: str, help_suffix: str = "") -> None:
    suffix = f" {help_suffix}" if help_suffix else ""
    parser.add_argument(f"{flag_prefix}-provider", dest=f"{dest_prefix}_provider", default=None, help=suffix.strip() or None)
    parser.add_argument(f"{flag_prefix}-model", dest=f"{dest_prefix}_model", default=None)
    parser.add_argument(f"{flag_prefix}-base-url", dest=f"{dest_prefix}_base_url", default=None)
    parser.add_argument(
        f"{flag_prefix}-api-key-env",
        dest=f"{dest_prefix}_api_key_env",
        default=None,
        help="env var NAME holding the key, never the key itself",
    )
    parser.add_argument(
        f"{flag_prefix}-extra-body",
        dest=f"{dest_prefix}_extra_body",
        default=None,
        help="JSON object merged into the request body",
    )


def resolve_llm_spec(override: dict | None, default: dict | None) -> dict:
    """Complete override wins as a block. Partial override does not mix with default."""
    if spec_complete(override):
        return normalize_spec(override)
    if spec_complete(default):
        return normalize_spec(default)
    return normalize_spec(override or {})


def resolved_plugin_llm(frame: Frame, override: dict | None) -> dict:
    return resolve_llm_spec(override, frame.default_llm)


def client_from_spec(spec: dict, timeout: float | None = None):
    if not spec_complete(spec):
        raise ValueError("LLM spec is missing provider or model")
    api_key = os.environ.get(spec["api_key_env"]) if spec.get("api_key_env") else None
    return get_client(
        spec["provider"],
        spec["model"],
        base_url=spec.get("base_url"),
        api_key=api_key,
        timeout=timeout,
        extra_body=spec.get("extra_body"),
    )


def sidecar_path(state_or_dir: str | Path) -> Path:
    path = Path(state_or_dir)
    if path.name == AGENT_LLM_FILENAME:
        return path
    if path.suffix == ".json":
        return path.parent / AGENT_LLM_FILENAME
    return path / AGENT_LLM_FILENAME


def load_sidecar(state_or_dir: str | Path) -> dict | None:
    path = sidecar_path(state_or_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def core_spec_from_sidecar(payload: dict | None) -> dict:
    if not payload:
        return {}
    return normalize_spec({k: payload.get(k) for k in SPEC_KEYS})


def write_sidecar(directory: str | Path, default_spec: dict, plugin_overrides: dict | None = None) -> Path:
    path = sidecar_path(directory)
    payload = {**normalize_spec(default_spec), "plugins": {}}
    for name, spec in (plugin_overrides or {}).items():
        if spec_complete(spec):
            payload["plugins"][name] = normalize_spec(spec)
    path.write_text(json.dumps(payload, indent=2))
    return path


def apply_sidecar_plugin_overrides(frame: Frame, payload: dict | None) -> None:
    plugins = (payload or {}).get("plugins") or {}
    cake_spec = plugins.get("cake")
    if spec_complete(cake_spec):
        blob = frame.plugins.setdefault("cake", {})
        if not spec_complete(blob.get("kernel_llm")):
            blob["kernel_llm"] = normalize_spec(cake_spec)
    llambo_spec = plugins.get("llambo")
    if spec_complete(llambo_spec):
        blob = frame.plugins.setdefault("llambo", {})
        if not spec_complete(blob.get("llm")):
            blob["llm"] = normalize_spec(llambo_spec)


def stamp_workdir(
    workdir: str | Path,
    default_spec: dict,
    plugin_overrides: dict | None = None,
) -> None:
    """Write ``agent_llm.json`` and, if ``state.json`` exists, fill empty LLM slots."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    payload_overrides = plugin_overrides or {}
    write_sidecar(workdir, default_spec, payload_overrides)
    state_path = workdir / "state.json"
    if not state_path.exists():
        return
    frame = Frame.load(str(state_path))
    if spec_complete(default_spec) and not spec_complete(frame.default_llm):
        frame.default_llm = normalize_spec(default_spec)
    apply_sidecar_plugin_overrides(frame, {"plugins": payload_overrides})
    frame.save(str(state_path))


def export_api_key(provider: str, api_key: str | None) -> None:
    """Put ``--api-key`` into the provider's standard env var so plugin subprocesses inherit it."""
    env_name = default_api_key_env(provider)
    if api_key and env_name:
        os.environ[env_name] = api_key
