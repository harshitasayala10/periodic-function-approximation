"""Ground-truth function f(x).

Replace the body of ``f`` with the function provided by Andromeda.
Nothing in the model, loss, or inference path should import this module.

Contract:
  - f is periodic: f(x + T) = f(x) for some unknown period T
  - f is bounded: |f(x)| <= C for some constant C
  - Domain: x can be any real number
"""

import math


def f(x: float) -> float:
    """Black-box periodic bounded function.

    Args:
        x: Input value, any real number.

    Returns:
        Bounded scalar output.
    """
    # Placeholder until the real function is provided.
    # Periodic with period 2π and range inside [-1.5, 1.5].
    # Two frequencies, so a single sine is not enough.
    return math.sin(x) + 0.5 * math.sin(2.0 * x)
