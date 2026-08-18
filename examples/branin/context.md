# Prior knowledge for this tuning task

Minimize the Branin function over a 2-D search space.

## Search space

| parameter | range | type | notes |
|---|---|---|---|
| `x1` | -5 .. 10 | float | |
| `x2` | 0 .. 15 | float | |

The evaluation command returns `{"y": <value>}`. Minimize `y`.

This is a smooth, well-behaved function with three known global minima. No other domain knowledge is available; treat it as a genuine
black box.
