#!/usr/bin/env python3
"""Black-box oracle for the Sara/lenz getting-started example: the 2-D
Branin function.

Usage: python3 eval.py '{"x1": 1.0, "x2": 2.0}'
Prints: {"y": <branin(x1, x2)>}
"""

import json
import math
import sys


def branin(x1: float, x2: float) -> float:
    a, b, c, r, s, t = 1.0, 5.1 / (4 * math.pi**2), 5.0 / math.pi, 6.0, 10.0, 1.0 / (8 * math.pi)
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * math.cos(x1) + s


def main() -> None:
    config = json.loads(sys.argv[1])
    y = branin(config["x1"], config["x2"])
    print(json.dumps({"y": y}))


if __name__ == "__main__":
    main()
