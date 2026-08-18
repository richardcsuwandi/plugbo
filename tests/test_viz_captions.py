from viz.captions import bench_label, classify_group, classify_relpath, experiment_caption


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


def test_classify_relpath_backend_sweep():
    tax = classify_relpath("hartmann6-compare/cake/sandbox_abc")
    assert tax["benchmark"] == "hartmann6"
    assert tax["benchmark_label"] == "Hartmann-6"
    assert tax["backend"] == "cake"
    assert tax["disclosure"] == "blind"
    assert tax["group"] == "hartmann6-compare"
    assert tax["condition"] == "cake"
    assert "blind" in tax["heading"]


def test_classify_relpath_sara_only_and_bolt():
    tax = classify_relpath("bolt_lora-compare/sara-lenz-cake/sandbox_fc186d26ab4797b9")
    assert tax["benchmark"] == "bolt_lora"
    assert tax["backend"] == "sara-cake"
    assert tax["disclosure"] == "revealed"
    assert "domain" in tax["heading"]
    assert "blind" not in tax["heading"]
    only = classify_relpath("bolt_lora-compare/sara-only/sandbox_x")
    assert only["backend"] == "sara-only"
    assert only["disclosure"] == "revealed"
    assert "Blind search" not in experiment_caption("bolt_lora-compare")
    assert "Domain LoRA" in experiment_caption("bolt_lora-compare")


def test_classify_relpath_disclosure_triangle():
    tax = classify_relpath("hartmann6-noblind-compare-cake/noblind-shift/sandbox_x")
    assert tax["benchmark"] == "hartmann6"
    assert tax["backend"] == "sara-cake"
    assert tax["disclosure"] == "shifted"


def test_classify_relpath_shifted_backend_sweep():
    tax = classify_relpath("hartmann6-noblind-compare-3config-shifted/sara-lenz/sandbox_x")
    assert tax["disclosure"] == "shifted"
    assert tax["backend"] == "sara-lenz"
    assert tax["axis"] == "backend"


def test_classify_group_headings():
    g = classify_group("ackley10-compare")
    assert g["benchmark"] == "ackley10"
    assert g["disclosure"] == "blind"
    assert g["backend"] is None
    d = classify_group("hartmann6-noblind-compare")
    assert d["axis"] == "disclosure"
    assert d["backend"] == "sara-lenz"


def test_classify_bolt_lora_followup_groups():
    domain = classify_group("bolt_lora-compare")
    assert domain["disclosure"] == "revealed"
    assert domain["heading"] == "BOLT LoRA HPO · domain context"
    g = classify_group("bolt_lora-generic-compare")
    assert g["benchmark"] == "bolt_lora"
    assert g["disclosure"] == "revealed"
    assert "generic" in g["heading"]
    m = classify_group("bolt_lora-misleading-compare")
    assert m["disclosure"] == "revealed"
    assert "misleading" in m["heading"]
    s = classify_group("bolt_lora-seed7-compare")
    assert s["benchmark"] == "bolt_lora"
    assert s["disclosure"] == "revealed"
    assert "seed 7" in s["heading"]
    assert "generic context" in experiment_caption("bolt_lora-generic-compare").lower()
    assert "seed 13" in experiment_caption("bolt_lora-seed13-compare")
