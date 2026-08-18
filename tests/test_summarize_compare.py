from benchmarks import summarize_compare


def test_summarize_condition_aggregates_repeated_seeds(monkeypatch, tmp_path):
    runs = [
        {
            "sandbox": "sandbox_a",
            "seed": 42,
            "trace": [1.0] * 24 + [0.5] * 76,
            "best_regret": 0.5,
            "n_evals": 100,
            "budget": 100,
            "status": "complete",
        },
        {
            "sandbox": "sandbox_b",
            "seed": 43,
            "trace": [2.0] * 49 + [0.25] * 51,
            "best_regret": 0.25,
            "n_evals": 100,
            "budget": 100,
            "status": "complete",
        },
    ]
    monkeypatch.setattr(summarize_compare, "all_condition_summaries", lambda _: runs)

    summary = summarize_compare.summarize_condition(tmp_path)

    assert summary["n_runs"] == 2
    assert summary["n_complete"] == 2
    assert summary["final_regret"]["median"] == 0.375
    assert summary["checkpoints"]["25"]["median"] == 1.25
    assert summary["checkpoints"]["100"]["n"] == 2
