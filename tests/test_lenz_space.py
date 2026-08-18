import pytest

from lenz.space import Encoder, RangeDim, SearchSpace, SpaceError


def test_range_and_choice_roundtrip():
    space = SearchSpace.from_json(
        {
            "x1": {"kind": "range", "lower": -5.0, "upper": 10.0},
            "n": {"kind": "range", "lower": 1, "upper": 8, "step": 1, "type": "int"},
            "color": {"kind": "choice", "values": ["red", "green", "blue"]},
        }
    )
    enc = Encoder(space)
    cfg = {"x1": 3.2, "n": 4, "color": "green"}
    x = enc.encode(cfg)
    back = enc.decode(x)
    assert back["color"] == "green"
    assert back["n"] == 4
    assert abs(back["x1"] - 3.2) < 1e-6


def test_log_scale_roundtrip():
    space = SearchSpace.from_json({"lr": {"kind": "range", "lower": 1e-4, "upper": 1e-1, "log_scale": True}})
    enc = Encoder(space)
    x = enc.encode({"lr": 1e-2})
    back = enc.decode(x)
    assert abs(back["lr"] - 1e-2) < 1e-9
    # domain bounds are in log-space
    assert enc.domain_bounds[0, 0].item() == pytest.approx(-9.210340371976182, abs=1e-6)


def test_invalid_range_raises():
    with pytest.raises(SpaceError):
        RangeDim(name="x", lower=1.0, upper=0.0)
    with pytest.raises(SpaceError):
        RangeDim(name="lr", lower=-1.0, upper=1.0, log_scale=True)


def test_missing_dim_raises():
    space = SearchSpace.from_json({"x": {"kind": "range", "lower": 0, "upper": 1}})
    with pytest.raises(SpaceError):
        SearchSpace.from_json({"x": {"kind": "bogus"}})
    with pytest.raises(SpaceError):
        space.validate_config_keys({})
    with pytest.raises(SpaceError):
        space.validate_config_keys({"x": 0.5, "y": 1.0})


def test_bounds_subset_validation():
    space = SearchSpace.from_json({"x": {"kind": "range", "lower": 0.0, "upper": 1.0}})
    enc = Encoder(space)
    ok = enc.encode_bounds({"x": [0.2, 0.8]})
    assert ok[0, 0].item() == pytest.approx(0.2)
    with pytest.raises(SpaceError):
        enc.encode_bounds({"x": [-0.5, 0.8]})


def test_radius_bounds_pins_choice_and_fixes_range():
    space = SearchSpace.from_json(
        {
            "x": {"kind": "range", "lower": 0.0, "upper": 1.0},
            "opt": {"kind": "choice", "values": ["a", "b", "c"]},
        }
    )
    enc = Encoder(space)
    incumbent = {"x": 0.5, "opt": "b"}
    bounds = enc.radius_bounds(incumbent, radius=0.1)
    x_sl = enc._col_slices["x"]
    assert bounds[0, x_sl][0].item() == pytest.approx(0.4)
    assert bounds[1, x_sl][0].item() == pytest.approx(0.6)
    opt_sl = enc._col_slices["opt"]
    # only "b"'s column should have nonzero upper bound (pinned at incumbent)
    assert bounds[1, opt_sl].tolist() == [0.0, 1.0, 0.0]

    with pytest.raises(SpaceError):
        enc.radius_bounds(incumbent, radius=0.0)
