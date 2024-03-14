"""fairtool: Toolbox for Fairness-aware Machine Learning.

``fairtool`` is a Python toolkit designed to help practitioners 
to understand, audit, mitigate, and evaluate the unfairness in 
both the input data and the produced models.

Subpackages
-----------
dataset
    Utilities to fetch and process fariness-related machine learing datasets.
inspection
    [Under Construction] Data and model fairness inspection methods.
metrics
    [Under Construction] Group and individual fairness evaluation metrics.
debias
    [Under Construction] Debiasing methods for fair machine learning.
"""

from . import dataset
from . import inspection
from . import metrics
from . import debias

__all__ = [
    "dataset",
    "inspection",
    "metrics",
    "debias",
]
