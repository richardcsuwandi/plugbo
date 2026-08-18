import pytest
import torch

from lenz.models import MIN_POINTS, ModelError, build_model_set
from lenz.space import Encoder, SearchSpace
from lenz.state import Frame, Objective, Shelf


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
