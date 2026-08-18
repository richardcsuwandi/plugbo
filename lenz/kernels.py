"""Kernel expression parsing: `"M5 + PER * LIN"` -> a GPyTorch composite
`Kernel`. Matches the CAKE paper's expression grammar, with a typed error
instead of a bare `except` so lenz/cake.py can distinguish a malformed
LLM-proposed expression from a real bug.
"""

from __future__ import annotations

import re

from gpytorch.kernels import Kernel, LinearKernel, MaternKernel, PeriodicKernel, RBFKernel, RQKernel, ScaleKernel

# CAKE's population is drawn from these six; M1 is parseable (matches the
# original gp.py's base-kernel dict) but not offered to the LLM by default.
DEFAULT_POPULATION = ["SE", "PER", "LIN", "RQ", "M3", "M5"]

MAX_NESTING = 50


class KernelParseError(ValueError):
    pass


def _base_kernels(d: int) -> dict[str, Kernel]:
    return {
        "SE": RBFKernel(ard_num_dims=d),
        "PER": PeriodicKernel(ard_num_dims=d),
        "LIN": LinearKernel(ard_num_dims=d),
        "RQ": RQKernel(ard_num_dims=d),
        "M1": MaternKernel(nu=0.5, ard_num_dims=d),
        "M3": MaternKernel(nu=1.5, ard_num_dims=d),
        "M5": MaternKernel(nu=2.5, ard_num_dims=d),
    }


def parse_kernel_expression(expression: str, d: int) -> Kernel:
    """Parses `"M5 + PER * LIN"` or `"(SE + PER) * RQ"` into a composite
    `ScaleKernel`. Raises `KernelParseError` on malformed or unknown-token
    input rather than crashing with a raw `KeyError`/`IndexError`.
    """
    if not expression or not expression.strip():
        raise KernelParseError("empty kernel expression")

    base_kernels = _base_kernels(d)

    def apply_operation(left: Kernel, op: str, right: Kernel) -> Kernel:
        if op == "+":
            return left + right
        if op == "*":
            return left * right
        raise KernelParseError(f"unknown operator '{op}'")

    def parse_subexpression(subexpr: str) -> Kernel:
        names = re.findall(r"[\w]+", subexpr)
        operators = re.findall(r"[+*]", subexpr)
        if not names:
            raise KernelParseError(f"no kernel names found in '{subexpr}'")
        if len(operators) != len(names) - 1:
            raise KernelParseError(f"malformed kernel expression '{subexpr}'")
        try:
            result = base_kernels[names[0]]
        except KeyError:
            raise KernelParseError(f"unknown base kernel '{names[0]}'") from None
        for i, op in enumerate(operators):
            try:
                right = base_kernels[names[i + 1]]
            except KeyError:
                raise KernelParseError(f"unknown base kernel '{names[i + 1]}'") from None
            result = apply_operation(result, op, right)
        return ScaleKernel(result)

    working = expression
    pattern = re.compile(r"\(([^()]+)\)")
    cache: dict[str, Kernel] = {}
    guard = 0
    while "(" in working:
        guard += 1
        if guard > MAX_NESTING:
            raise KernelParseError(f"kernel expression too deeply nested: '{expression}'")
        matches = pattern.findall(working)
        if not matches:
            raise KernelParseError(f"unbalanced parentheses in '{expression}'")
        for subexpr in matches:
            if subexpr not in cache:
                sub_kernel = parse_subexpression(subexpr)
                cache[subexpr] = sub_kernel
                base_kernels[f"SubKernel{len(base_kernels)}"] = sub_kernel
            placeholder = f"SubKernel{len(base_kernels) - 1}"
            working = working.replace(f"({subexpr})", placeholder, 1)

    return parse_subexpression(working)
