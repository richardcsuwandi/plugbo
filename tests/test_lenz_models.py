import pytest
import torch

from lenz.models import ModelError, build_model_set
from lenz.space import Encoder, SearchSpace
from lenz.state import Frame, Objective, Shelf
from lenz import cake


def _frame(minimize: bool) -> Frame:
    space = SearchSpace.from_json({"x": {"kind": "range", "lower": 0.0, "upper": 1.0}})
    shelf = Shelf(objectives=[Objective(metric="y", minimize=minimize)])
    frame = Frame(space=space, shelf=shelf)
    for x, y in [(0.1, 1.0), (0.5, 5.0), (0.9, 2.0)]:
        frame.submit({"x": x}, {"y": y})
    return frame


def test_needs_minimum_points():
    space = SearchSpace.from_json({"x": {"kind": "range", "lower": 0.0, "upper": 1.0}})
    shelf = Shelf(objectives=[Objective(metric="y", minimize=True)])
    frame = Frame(space=space, shelf=shelf)
    frame.submit({"x": 0.5}, {"y": 1.0})
    with pytest.raises(ModelError):
        build_model_set(frame, Encoder(frame.space))


def test_sign_convention_maximize():
    frame = _frame(minimize=False)
    ms = build_model_set(frame, Encoder(frame.space))
    assert ms.objective_sign["y"] == 1.0
    # best raw y (5.0 at x=0.5) should also be the best in sign-adjusted space
    best_idx = int(torch.argmax(ms.Y_raw["y"] * ms.objective_sign["y"]))
    assert ms.X[best_idx, 0].item() == pytest.approx(0.5)


def test_sign_convention_minimize():
    frame = _frame(minimize=True)
    ms = build_model_set(frame, Encoder(frame.space))
    assert ms.objective_sign["y"] == -1.0
    # under minimize, x=0.1 (y=1.0) is best -> should have the largest sign-adjusted value
    best_idx = int(torch.argmax(ms.Y_raw["y"] * ms.objective_sign["y"]))
    assert ms.X[best_idx, 0].item() == pytest.approx(0.1)


def test_missing_metric_raises():
    space = SearchSpace.from_json({"x": {"kind": "range", "lower": 0.0, "upper": 1.0}})
    shelf = Shelf(objectives=[Objective(metric="y", minimize=True)])
    frame = Frame(space=space, shelf=shelf)
    frame.submit({"x": 0.1}, {"other": 1.0})
    frame.submit({"x": 0.9}, {"other": 2.0})
    with pytest.raises(ModelError):
        build_model_set(frame, Encoder(frame.space))


def _mixed_space() -> SearchSpace:
    return SearchSpace.from_json(
        {
            "lr": {"kind": "range", "lower": 0.0, "upper": 1.0},
            "batch": {"kind": "range", "lower": 2, "upper": 4, "type": "int"},
            "mod": {"kind": "choice", "values": [0, 1, 2, 3]},
        }
    )


def test_model_set_fits_on_continuous_x_gp_not_projected_config():
    space = _mixed_space()
    encoder = Encoder(space)
    shelf = Shelf(objectives=[Objective(metric="y", minimize=False)])
    frame = Frame(space=space, shelf=shelf)
    # batch 2.4 projects to 2; soft one-hot argmax is mod=1.
    x_gp_a = [0.31, 2.4, 0.10, 0.70, 0.15, 0.05]
    x_gp_b = [0.82, 3.6, 0.05, 0.10, 0.80, 0.05]
    cfg_a = encoder.decode(torch.tensor(x_gp_a, dtype=torch.double))
    cfg_b = encoder.decode(torch.tensor(x_gp_b, dtype=torch.double))
    assert cfg_a["batch"] == 2
    assert cfg_a["mod"] == 1
    assert cfg_b["batch"] == 4
    assert cfg_b["mod"] == 2
    frame.submit(cfg_a, {"y": 1.0}, x_gp=x_gp_a)
    frame.submit(cfg_b, {"y": 2.0}, x_gp=x_gp_b)
    ms = build_model_set(frame, encoder)
    assert ms.X[0].tolist() == pytest.approx(x_gp_a, abs=1e-9)
    assert ms.X[1].tolist() == pytest.approx(x_gp_b, abs=1e-9)
    encoded_a = encoder.encode(cfg_a)
    assert encoded_a[1].item() == pytest.approx(2.0)
    assert ms.X[0, 1].item() == pytest.approx(2.4)


def test_submit_recovers_x_gp_from_suggest():
    from lenz.optimize import suggest

    space = _mixed_space()
    encoder = Encoder(space)
    shelf = Shelf(objectives=[Objective(metric="y", minimize=False)], seed=0)
    frame = Frame(space=space, shelf=shelf)
    out = suggest(frame, encoder, q=1)
    assert "x_gp" in out[0]
    assert len(out[0]["x_gp"]) == encoder.d
    trial = frame.submit(out[0]["config"], {"y": 0.5})
    assert trial.x_gp == pytest.approx(out[0]["x_gp"])


def test_suggest_skips_unfittable_cake_kernels_and_uses_next_viable():
    from lenz.optimize import suggest

    space = _mixed_space()
    shelf = Shelf(objectives=[Objective(metric="y", minimize=False)], surrogate="cake", acqf="noisy_logei")
    frame = Frame(space=space, shelf=shelf)
    encoder = Encoder(space)
    rng = torch.Generator().manual_seed(0)
    for _ in range(10):
        x = torch.rand(encoder.d, generator=rng, dtype=torch.double)
        x = torch.clamp(x, encoder.domain_bounds[0], encoder.domain_bounds[1])
        x[1] = 2.0 + float(x[1].item()) * 2.0
        cfg = encoder.decode(x)
        frame.submit(cfg, {"y": float(x[0].item()) + cfg["batch"]}, x_gp=x.tolist())
    cake.state(frame)["kernel_populations"]["y"] = [
        {"expression": "PER", "bic": float("inf"), "generation": 1},
        {"expression": "LIN", "bic": float("inf"), "generation": 1},
        {"expression": "M5", "bic": 12.0, "generation": 1},
    ]
    out = suggest(frame, encoder, q=1)
    assert "config" in out[0]
    assert set(out[0]["config"]) == {"lr", "batch", "mod"}
    assert "x_gp" in out[0]


def test_suggest_does_not_swap_in_matern_when_cake_has_no_fittable_kernel():
    from lenz.optimize import suggest

    space = _mixed_space()
    shelf = Shelf(
        objectives=[Objective(metric="y", minimize=False)],
        surrogate="cake",
        acqf="noisy_logei",
        seed=1,
    )
    frame = Frame(space=space, shelf=shelf)
    encoder = Encoder(space)
    rng = torch.Generator().manual_seed(1)
    for _ in range(encoder.d + 2):
        x = torch.rand(encoder.d, generator=rng, dtype=torch.double)
        x = torch.clamp(x, encoder.domain_bounds[0], encoder.domain_bounds[1])
        x[1] = 2.0 + float(x[1].item()) * 2.0
        cfg = encoder.decode(x)
        frame.submit(cfg, {"y": 1.0}, x_gp=x.tolist())
    cake.state(frame)["kernel_populations"]["y"] = [
        {"expression": "NOT_A_KERNEL", "bic": float("inf"), "generation": 1},
        {"expression": "ALSO_INVALID", "bic": float("inf"), "generation": 1},
    ]
    out = suggest(frame, encoder, q=1)
    # Honest fallback: Sobol, not a hidden default Matérn under the cake label.
    assert out[0]["acqf"] == "sobol"
