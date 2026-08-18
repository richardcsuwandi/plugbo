import pytest

from lenz.kernels import DEFAULT_POPULATION, KernelParseError, parse_kernel_expression


@pytest.mark.parametrize("name", DEFAULT_POPULATION)
def test_all_default_population_kernels_parse(name):
    k = parse_kernel_expression(name, d=3)
    assert type(k).__name__ == "ScaleKernel"


def test_composite_expression():
    k = parse_kernel_expression("M5 + PER * LIN", d=2)
    assert type(k).__name__ == "ScaleKernel"


def test_nested_parentheses():
    k = parse_kernel_expression("(SE + PER) * RQ", d=2)
    assert type(k).__name__ == "ScaleKernel"


def test_deeply_nested_parentheses():
    k = parse_kernel_expression("((SE + PER) * RQ) + M3", d=2)
    assert type(k).__name__ == "ScaleKernel"


@pytest.mark.parametrize(
    "bad_expr",
    ["", "   ", "SE +", "BOGUS", "SE PER", "(SE + PER", "SE ++ PER", "SE + BOGUS"],
)
def test_malformed_expressions_raise(bad_expr):
    with pytest.raises(KernelParseError):
        parse_kernel_expression(bad_expr, d=2)


def test_m1_parseable_but_not_in_default_population():
    # M1 (Matern-0.5) is a valid base kernel the parser supports, matching CAKE's
    # gp.py, even though it isn't offered to the LLM by default.
    assert "M1" not in DEFAULT_POPULATION
    k = parse_kernel_expression("M1", d=1)
    assert type(k).__name__ == "ScaleKernel"


def test_stray_trailing_punctuation_is_tolerated():
    # A trailing unmatched ")" with no corresponding "(" never enters the paren-handling
    # loop, so it's silently ignored by the `\w+` name extraction -- same leniency as
    # CAKE's original regex-based parser. Only a genuine open/close mismatch is rejected.
    k = parse_kernel_expression("SE + PER)", d=2)
    assert type(k).__name__ == "ScaleKernel"
