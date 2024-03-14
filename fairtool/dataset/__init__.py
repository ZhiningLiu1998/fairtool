"""
The :mod:`fairtool.dataset` provides utilities to fetch and process 
fariness-related machine learing datasets.
"""

from ._fetch_adult import fetch_adult

__all__ = [
    "fetch_adult",
]
