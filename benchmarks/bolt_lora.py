"""BOLT LoRA HPO emulator (Qwen3-8B LoRA fine-tuning surrogate).

Offline MLP emulator from Chew et al.'s BOLT suite, hosted at
`chewwt/hpo_qwen8b_emulator`. The 7-D mixed space and the encoding applied
before the MLP match `bolt.problems.hpo.HPO` in the GRAPE/BOLT repo so
reported scores stay comparable. Weights are downloaded on first evaluation
and cached by `huggingface_hub`. Transient TLS drops against huggingface.co
are retried, then `https://hf-mirror.com` is tried. Override with `HF_ENDPOINT`
or `BOLT_HF_ENDPOINT`.

Optional deps: `huggingface_hub`, `safetensors` (see `pip install -e '.[bolt]'`).
"""

from __future__ import annotations

import csv
import json
import os
import ssl
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

HF_REPO = "chewwt/hpo_qwen8b_emulator"
# huggingface.co often drops TLS mid-download (SSLEOF). Honor HF_ENDPOINT if
# set; otherwise retry the default hub, then the public hf-mirror.com replica.
HF_MIRROR = "https://hf-mirror.com"
_DOWNLOAD_ATTEMPTS = 3

# Order matches BOLT's HPO._bounds / discrete_inds / categorical_inds.
PARAM_ORDER = (
    "lr",
    "batch",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "lora_layers",
    "lora_target",
)
BOUNDS = [
    (0.0, 1.0),  # lr
    (2, 4),  # batch
    (2, 5),  # lora rank
    (2, 5),  # lora alpha
    (0.0, 1.0),  # lora dropout
    (1, 30),  # lora layers
    (0, 3),  # lora target module (categorical index)
]
DISCRETE_INDS = (1, 2, 3, 5)
CATEGORICAL_IND = 6
N_TARGET_CLASSES = 4

# Empirically reported in BOLT's HPO class. Used as a gap reference, not a
# closed-form optimum. Must never appear in agent-visible context.md.
F_OPT = 0.34647
X_OPT = [0.31100, 2, 4, 2, 0.87056, 30, 1]

SPACE: dict[str, dict[str, Any]] = {
    "lr": {"kind": "range", "lower": 0.0, "upper": 1.0, "type": "float"},
    "batch": {"kind": "range", "lower": 2, "upper": 4, "type": "int", "step": 1},
    "lora_rank": {"kind": "range", "lower": 2, "upper": 5, "type": "int", "step": 1},
    "lora_alpha": {"kind": "range", "lower": 2, "upper": 5, "type": "int", "step": 1},
    "lora_dropout": {"kind": "range", "lower": 0.0, "upper": 1.0, "type": "float"},
    "lora_layers": {"kind": "range", "lower": 1, "upper": 30, "type": "int", "step": 1},
    "lora_target": {"kind": "choice", "values": [0, 1, 2, 3]},
}

_STATE: dict[str, Any] | None = None


class _FeatureMLP(nn.Module):
    """Three-layer MLP + LayerNorm, matching BOLT's FeatureNet."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(int(input_dim), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def _require_bolt_deps() -> None:
    try:
        import huggingface_hub  # noqa: F401
        import safetensors  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "bolt_lora requires huggingface_hub and safetensors. "
            "Install with: pip install -e '.[bolt]'"
        ) from e


def _hub_endpoints() -> list[str | None]:
    explicit = os.environ.get("HF_ENDPOINT") or os.environ.get("BOLT_HF_ENDPOINT")
    if explicit:
        return [explicit.rstrip("/")]
    return [None, HF_MIRROR]


def _is_transient_hub_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in {"RepositoryNotFoundError", "EntryNotFoundError", "RevisionNotFoundError", "GatedRepoError"}:
        return False
    text = str(exc).lower()
    needles = (
        "ssl",
        "eof",
        "connection",
        "timed out",
        "timeout",
        "reset",
        "temporarily",
        "max retries",
        "429",
        "502",
        "503",
        "504",
        "chunked",
    )
    if any(n in text for n in needles):
        return True
    return isinstance(exc, (OSError, TimeoutError, ConnectionError, ssl.SSLError))


def _hf_download(filename: str) -> str:
    """`hf_hub_download` with a local cache hit first, then resume, SSL retries,
    and an optional mirror fallback.
    """
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(repo_id=HF_REPO, filename=filename, local_files_only=True)
    except Exception:
        pass

    last_err: BaseException | None = None
    for endpoint in _hub_endpoints():
        label = endpoint or "huggingface.co"
        for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
            try:
                kwargs: dict[str, Any] = {
                    "repo_id": HF_REPO,
                    "filename": filename,
                    "etag_timeout": 30.0,
                }
                if endpoint is not None:
                    kwargs["endpoint"] = endpoint
                return hf_hub_download(**kwargs)
            except Exception as e:
                last_err = e
                if not _is_transient_hub_error(e) or attempt == _DOWNLOAD_ATTEMPTS:
                    break
                delay = min(2 ** (attempt - 1), 16)
                print(
                    f"bolt_lora: {filename} from {label} failed ({type(e).__name__}); "
                    f"retry {attempt}/{_DOWNLOAD_ATTEMPTS} in {delay}s",
                    flush=True,
                )
                time.sleep(delay)
        print(f"bolt_lora: giving up on {label} for {filename}", flush=True)
    hint = (
        "Set HF_ENDPOINT=https://hf-mirror.com (or BOLT_HF_ENDPOINT) if huggingface.co "
        "is unreachable, then retry."
    )
    raise RuntimeError(f"failed to download {HF_REPO}/{filename}. {hint}") from last_err


def _load() -> dict[str, Any]:
    global _STATE
    if _STATE is not None:
        return _STATE
    _require_bolt_deps()
    from safetensors.torch import load_file

    model_path = _hf_download("model.safetensors")
    config_path = _hf_download("config.json")
    csv_path = _hf_download("model_standardize.csv")
    with open(config_path) as f:
        model_config = json.load(f)
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    y_mean = torch.tensor([float(r["y_mean"]) for r in rows], dtype=torch.double)
    y_std = torch.tensor([float(r["y_std"]) for r in rows], dtype=torch.double)

    model = _FeatureMLP(
        input_dim=int(model_config["input_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        output_dim=int(model_config["output_dim"]),
    )
    model.load_state_dict(load_file(model_path))
    model.eval()
    _STATE = {"model": model, "y_mean": y_mean, "y_std": y_std}
    return _STATE


def encode_features(x: torch.Tensor) -> torch.Tensor:
    """Map a (N, 7) raw-parameter tensor to the emulator's 11-D feature vector."""
    x_copy = x.clone()
    for i in DISCRETE_INDS:
        i_min, i_max = BOUNDS[i]
        x_copy[:, i] = (x_copy[:, i] - i_min) / (i_max - i_min)
    target = x_copy[:, CATEGORICAL_IND].long().clamp(0, N_TARGET_CLASSES - 1)
    one_hot = F.one_hot(target, num_classes=N_TARGET_CLASSES).to(dtype=x_copy.dtype)
    ones = torch.ones(x_copy.shape[0], 1, dtype=x_copy.dtype, device=x_copy.device)
    return torch.cat([x_copy[:, :CATEGORICAL_IND], one_hot, ones], dim=1)


def evaluate_vector(x: list[float]) -> float:
    """Score one 7-D configuration (BOLT raw parameter order). Higher is better."""
    if len(x) != 7:
        raise ValueError(f"bolt_lora expects 7 parameters, got {len(x)}")
    state = _load()
    xt = torch.tensor([x], dtype=torch.double)
    xt[0, CATEGORICAL_IND] = int(round(float(xt[0, CATEGORICAL_IND].item())))
    for i in DISCRETE_INDS:
        xt[0, i] = int(round(float(xt[0, i].item())))
    feats = encode_features(xt)
    model = state["model"]
    model_dtype = next(model.parameters()).dtype
    with torch.no_grad():
        y_st = model(feats.to(dtype=model_dtype)).to(dtype=torch.double)
    y = y_st.squeeze() * state["y_std"] + state["y_mean"]
    return float(y.reshape(-1)[0].item())


def decode_config(payload: dict, config: dict) -> list:
    """Turn an agent-facing config dict into the 7-D raw vector using oracle payload."""
    x = []
    for dim in payload["dims"]:
        raw = config[dim["name"]]
        if dim["kind"] == "choice":
            agent_vals = dim["agent_values"]
            if raw not in agent_vals and isinstance(raw, float) and raw == int(raw):
                raw = int(raw)
            idx = agent_vals.index(raw)
            x.append(dim["true_values"][idx])
        elif dim.get("type") == "int":
            x.append(int(round(float(raw))))
        else:
            x.append(float(raw))
    return x


def evaluate_payload(payload: dict, config: dict) -> dict:
    return {"y": evaluate_vector(decode_config(payload, config))}


def oracle_payload(param_names: list[str], choice_values: dict, space: dict) -> dict:
    """JSON-serializable payload baked into the generated oracle script."""
    dims = []
    for i, true_name in enumerate(space):
        dim = space[true_name]
        agent_name = param_names[i]
        entry = {
            "name": agent_name,
            "kind": dim["kind"],
            "type": dim.get("type"),
            "true_values": dim.get("values"),
            "agent_values": choice_values.get(agent_name, dim.get("values")),
        }
        dims.append(entry)
    return {"dims": dims}
