"""Search-space specification and encoding to/from a continuous GP representation.

A `SearchSpace` holds the *original* domain declared at `create` time. Active
(persistently narrowed) bounds are tracked separately in the state's shelf and
must always be validated as a subset of this original domain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Union

import torch

DTYPE = torch.double


class SpaceError(ValueError):
    pass


@dataclass
class RangeDim:
    name: str
    lower: float
    upper: float
    step: float | None = None
    type: Literal["float", "int"] = "float"
    log_scale: bool = False

    def __post_init__(self) -> None:
        if self.lower >= self.upper:
            raise SpaceError(f"dimension '{self.name}': lower must be < upper")
        if self.log_scale and self.lower <= 0:
            raise SpaceError(f"dimension '{self.name}': log_scale requires lower > 0")

    def to_json(self) -> dict:
        d: dict[str, Any] = {"kind": "range", "lower": self.lower, "upper": self.upper, "type": self.type}
        if self.step is not None:
            d["step"] = self.step
        if self.log_scale:
            d["log_scale"] = True
        return d


@dataclass
class ChoiceDim:
    name: str
    values: list
    ordered: bool = False

    def __post_init__(self) -> None:
        if len(self.values) < 2:
            raise SpaceError(f"dimension '{self.name}': choice requires at least 2 values")

    def to_json(self) -> dict:
        return {"kind": "choice", "values": self.values, "ordered": self.ordered}


Dim = Union[RangeDim, ChoiceDim]


def _dim_from_json(name: str, spec: dict) -> Dim:
    kind = spec.get("kind")
    if kind == "range":
        if "lower" not in spec or "upper" not in spec:
            raise SpaceError(f"range dimension '{name}' requires 'lower' and 'upper'")
        return RangeDim(
            name=name,
            lower=float(spec["lower"]),
            upper=float(spec["upper"]),
            step=float(spec["step"]) if spec.get("step") is not None else None,
            type=spec.get("type", "float"),
            log_scale=bool(spec.get("log_scale", False)),
        )
    if kind == "choice":
        if "values" not in spec:
            raise SpaceError(f"choice dimension '{name}' requires 'values'")
        return ChoiceDim(name=name, values=list(spec["values"]), ordered=bool(spec.get("ordered", False)))
    raise SpaceError(f"dimension '{name}': unknown kind '{kind}' (expected 'range' or 'choice')")


@dataclass
class SearchSpace:
    dims: dict[str, Dim] = field(default_factory=dict)

    @classmethod
    def from_json(cls, space_json: dict) -> "SearchSpace":
        if not space_json:
            raise SpaceError("space must declare at least one dimension")
        return cls({name: _dim_from_json(name, spec) for name, spec in space_json.items()})

    def to_json(self) -> dict:
        return {name: dim.to_json() for name, dim in self.dims.items()}

    @property
    def names(self) -> list[str]:
        return list(self.dims.keys())

    def __contains__(self, name: str) -> bool:
        return name in self.dims

    def validate_config_keys(self, config: dict) -> None:
        missing = set(self.dims) - set(config)
        extra = set(config) - set(self.dims)
        if missing:
            raise SpaceError(f"config missing dimensions: {sorted(missing)}")
        if extra:
            raise SpaceError(f"config has unknown dimensions: {sorted(extra)}")


class Encoder:
    """Maps between raw configs (JSON dicts) and a continuous GP-space tensor.

    Range dims -> 1 column (log-transformed if `log_scale`).
    Ordered choice dims -> 1 column (treated as an integer index range).
    Unordered choice dims -> one-hot columns, one per value.
    """

    def __init__(self, space: SearchSpace):
        self.space = space
        self._col_names: list[str] = []  # GP-space column -> owning dim name
        self._col_slices: dict[str, slice] = {}
        lowers: list[float] = []
        uppers: list[float] = []
        col = 0
        for name, dim in space.dims.items():
            if isinstance(dim, RangeDim):
                lo, hi = self._range_to_gp(dim, dim.lower), self._range_to_gp(dim, dim.upper)
                lowers.append(lo)
                uppers.append(hi)
                self._col_slices[name] = slice(col, col + 1)
                self._col_names.append(name)
                col += 1
            elif dim.ordered:
                lowers.append(0.0)
                uppers.append(float(len(dim.values) - 1))
                self._col_slices[name] = slice(col, col + 1)
                self._col_names.append(name)
                col += 1
            else:
                k = len(dim.values)
                lowers.extend([0.0] * k)
                uppers.extend([1.0] * k)
                self._col_slices[name] = slice(col, col + k)
                self._col_names.extend([name] * k)
                col += k
        self.d = col
        self.domain_bounds = torch.tensor([lowers, uppers], dtype=DTYPE)

    @staticmethod
    def _range_to_gp(dim: RangeDim, raw: float) -> float:
        return math.log(raw) if dim.log_scale else raw

    @staticmethod
    def _range_from_gp(dim: RangeDim, gp_val: float) -> float:
        raw = math.exp(gp_val) if dim.log_scale else gp_val
        raw = min(max(raw, dim.lower), dim.upper)
        if dim.step:
            n = round((raw - dim.lower) / dim.step)
            raw = dim.lower + n * dim.step
            raw = min(max(raw, dim.lower), dim.upper)
        if dim.type == "int":
            raw = round(raw)
        return raw

    def encode(self, config: dict) -> torch.Tensor:
        self.space.validate_config_keys(config)
        x = torch.zeros(self.d, dtype=DTYPE)
        for name, dim in self.space.dims.items():
            sl = self._col_slices[name]
            if isinstance(dim, RangeDim):
                x[sl] = self._range_to_gp(dim, float(config[name]))
            elif dim.ordered:
                x[sl] = float(dim.values.index(config[name]))
            else:
                idx = dim.values.index(config[name])
                onehot = torch.zeros(len(dim.values), dtype=DTYPE)
                onehot[idx] = 1.0
                x[sl] = onehot
        return x

    def decode(self, x: torch.Tensor) -> dict:
        config: dict[str, Any] = {}
        for name, dim in self.space.dims.items():
            sl = self._col_slices[name]
            if isinstance(dim, RangeDim):
                config[name] = self._range_from_gp(dim, float(x[sl][0]))
            elif dim.ordered:
                idx = int(round(float(x[sl][0])))
                idx = min(max(idx, 0), len(dim.values) - 1)
                config[name] = dim.values[idx]
            else:
                idx = int(torch.argmax(x[sl]).item())
                config[name] = dim.values[idx]
        return config

    def encode_bounds(self, bounds: dict[str, list[float]] | None) -> torch.Tensor:
        """Build a (2, d) GP-space bounds tensor from a partial raw-units bounds dict.

        Dimensions not present in `bounds` default to the full original domain.
        Only RangeDim entries may appear in `bounds`.
        """
        lo = self.domain_bounds[0].clone()
        hi = self.domain_bounds[1].clone()
        if not bounds:
            return torch.stack([lo, hi])
        for name, (b_lo, b_hi) in bounds.items():
            if name not in self.space.dims:
                raise SpaceError(f"bounds reference unknown dimension '{name}'")
            dim = self.space.dims[name]
            if not isinstance(dim, RangeDim):
                raise SpaceError(f"bounds on '{name}': only range dimensions can be bounded")
            sl = self._col_slices[name]
            gp_lo = self._range_to_gp(dim, float(b_lo))
            gp_hi = self._range_to_gp(dim, float(b_hi))
            if gp_lo > gp_hi:
                gp_lo, gp_hi = gp_hi, gp_lo
            dom_lo, dom_hi = float(self.domain_bounds[0, sl][0]), float(self.domain_bounds[1, sl][0])
            eps = 1e-9 * max(1.0, abs(dom_hi - dom_lo))
            if gp_lo < dom_lo - eps or gp_hi > dom_hi + eps:
                raise SpaceError(f"bounds on '{name}': not a subset of the original domain")
            lo[sl] = gp_lo
            hi[sl] = gp_hi
        return torch.stack([lo, hi])

    def bounds_to_json(self, bounds: torch.Tensor) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for name, dim in self.space.dims.items():
            if not isinstance(dim, RangeDim):
                continue
            sl = self._col_slices[name]
            lo = self._range_from_gp_unrounded(dim, float(bounds[0, sl][0]))
            hi = self._range_from_gp_unrounded(dim, float(bounds[1, sl][0]))
            out[name] = [lo, hi]
        return out

    @staticmethod
    def _range_from_gp_unrounded(dim: RangeDim, gp_val: float) -> float:
        return math.exp(gp_val) if dim.log_scale else gp_val

    def radius_bounds(self, incumbent: dict, radius: float, per_dim: dict | None = None) -> torch.Tensor:
        """Bounds for `suggest --around`: a fractional radius of each domain width
        around the incumbent, clipped to the original domain. `per_dim` overrides
        (numeric radius, {"fix": value}, or a list restricting a choice) win per-dimension.
        """
        if not (0.0 < radius <= 1.0):
            raise SpaceError("radius must be in (0, 1]")
        x0 = self.encode(incumbent)
        lo = self.domain_bounds[0].clone()
        hi = self.domain_bounds[1].clone()
        for name, dim in self.space.dims.items():
            sl = self._col_slices[name]
            override = (per_dim or {}).get(name)
            if isinstance(dim, RangeDim):
                dom_lo, dom_hi = float(self.domain_bounds[0, sl][0]), float(self.domain_bounds[1, sl][0])
                width = dom_hi - dom_lo
                if isinstance(override, dict) and "fix" in override:
                    v = self._range_to_gp(dim, float(override["fix"]))
                    lo[sl], hi[sl] = v, v
                else:
                    r = float(override) if isinstance(override, (int, float)) else radius
                    center = float(x0[sl][0])
                    lo[sl] = max(dom_lo, center - r * width)
                    hi[sl] = min(dom_hi, center + r * width)
            else:
                if isinstance(override, list):
                    idxs = [dim.values.index(v) for v in override]
                elif isinstance(override, dict) and "fix" in override:
                    idxs = [dim.values.index(override["fix"])]
                elif dim.ordered:
                    idxs = None  # pin at incumbent below
                else:
                    idxs = [int(torch.argmax(x0[sl]).item())]  # pin unordered choice at incumbent
                if dim.ordered:
                    center = float(x0[sl][0])
                    if idxs is None:
                        lo[sl], hi[sl] = center, center
                    else:
                        lo[sl], hi[sl] = min(idxs), max(idxs)
                else:
                    mask = torch.zeros(len(dim.values), dtype=DTYPE)
                    for i in idxs:
                        mask[i] = 1.0
                    lo[sl] = 0.0
                    hi[sl] = mask
        return torch.stack([lo, hi])
