"""
The :mod:`fairtool.metrics` provides metrics for measuring 
group- and/or individual-level (un)fairness.
"""

from ._group_fairness import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    true_positive_rate,
    false_positive_rate,
    selection_rate,
)

__all__ = [
    "demographic_parity_difference",
    "demographic_parity_ratio",
    "equalized_odds_difference",
    "equalized_odds_ratio",
    "true_positive_rate",
    "false_positive_rate",
    "selection_rate",
]
