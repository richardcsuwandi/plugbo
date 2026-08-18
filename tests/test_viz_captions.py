from viz.captions import bench_label, experiment_caption


def test_bench_label_known_and_gp_sample():
    assert bench_label("hartmann6") == "Hartmann-6"
    assert bench_label("gp_sample6") == "GP sample (6D)"
    assert bench_label("ackley10") == "Ackley-10"


def test_experiment_caption_layouts():
    assert "Blind search" in experiment_caption("hartmann6-compare")
    assert "Fixed vanilla BO" in experiment_caption("hartmann6-noblind-compare-vanilla")
    assert "Fixed Sara+Lenz:" in experiment_caption("gp_sample6-noblind-compare")
    assert "Identity revealed, shifted optimum" in experiment_caption(
        "hartmann6-noblind-compare-3config-shifted"
    )
