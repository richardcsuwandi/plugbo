"""Seeded Sobol warmup: same seed replays the same initial design."""

import json
import subprocess
import sys

from lenz.optimize import sobol_candidates, suggest
from lenz.space import Encoder, SearchSpace
from lenz.state import Frame, Objective, Shelf


def _frame(seed=42):
    space = SearchSpace.from_json(
        {
            "x1": {"kind": "range", "lower": 0.0, "upper": 1.0},
            "x2": {"kind": "range", "lower": 0.0, "upper": 1.0},
        }
    )
    shelf = Shelf(objectives=[Objective(metric="y", minimize=True)], seed=seed)
    return Frame(space=space, shelf=shelf), Encoder(space)


def test_seeded_sobol_is_deterministic():
    space = SearchSpace.from_json({"x": {"kind": "range", "lower": 0.0, "upper": 1.0}})
    enc = Encoder(space)
    bounds = enc.encode_bounds({})
    a, _ = sobol_candidates(enc, bounds, q=5, seed=7, skip=0)
    b, _ = sobol_candidates(enc, bounds, q=5, seed=7, skip=0)
    assert a == b
    c, _ = sobol_candidates(enc, bounds, q=5, seed=8, skip=0)
    assert a != c


def test_seeded_sequential_q1_matches_batch():
    space = SearchSpace.from_json(
        {
            "x1": {"kind": "range", "lower": 0.0, "upper": 1.0},
            "x2": {"kind": "range", "lower": 0.0, "upper": 1.0},
        }
    )
    enc = Encoder(space)
    bounds = enc.encode_bounds({})
    batch, skip = sobol_candidates(enc, bounds, q=4, seed=42, skip=0)
    assert skip == 4
    sequential, skip2 = [], 0
    for _ in range(4):
        pts, skip2 = sobol_candidates(enc, bounds, q=1, seed=42, skip=skip2)
        sequential.extend(pts)
    assert sequential == batch


def test_suggest_advances_sobol_drawn():
    frame, encoder = _frame(seed=42)
    first = suggest(frame, encoder, q=3)
    assert all(p["acqf"] == "sobol" for p in first)
    assert frame.shelf.sobol_drawn == 3
    second = suggest(frame, encoder, q=2)
    assert frame.shelf.sobol_drawn == 5
    assert first[0]["config"] != second[0]["config"]


def test_create_refuses_to_overwrite_existing_state(tmp_path):
    state = tmp_path / "state.json"
    space = json.dumps({"x": {"kind": "range", "lower": 0.0, "upper": 1.0}})
    cmd = [
        sys.executable,
        "-m",
        "lenz.cli",
        "create",
        "--state",
        str(state),
        "--space",
        space,
        "--objectives",
        json.dumps({"y": "minimize"}),
    ]
    first = subprocess.run(cmd, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)["ok"]
    second = subprocess.run(cmd, capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    payload = json.loads(second.stdout)
    assert payload["ok"] is False
    assert "already exists" in payload["error"]
    forced = subprocess.run([*cmd, "--force"], capture_output=True, text=True)
    assert json.loads(forced.stdout)["ok"] is True
