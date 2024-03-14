# Copyright (c) fairtool contributors.
# Distributed under the terms of the MIT License.
# Initially adapted from fairlearn (https://github.com/fairlearn/fairlearn)

"""
Utilities for fetching & processing fair machine learning dataset.
"""

# %%

LOCAL_DEBUG = True

if not LOCAL_DEBUG:
    from . import fetch_adult
else:  # pragma: no cover
    # For local debugging purposes
    import sys

    sys.path.append("..")
    from dataset._fetch_adult import fetch_adult


def load_data(*, dataname: str, cache=True, data_home=None, return_X_y=False):
    """
    Load the specified dataset. We process all the datasets as pandas DataFrame.

    Parameters
    ----------
    dataname : str
        The name of the dataset to load.
    cache : bool, default=True
        Whether to cache the downloaded datasets into data_home.
    data_home : str, default=None
        Specify another download and cache folder for the datasets.
        By default, all fairtool datasets are stored in '~/.fairtool-data'
        subfolders.
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.

    Returns
    -------
    dataset : :obj:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : ndarray, shape (48842, 14)
            Each row corresponding to the 14 feature values in order.
            If ``as_frame`` is True, ``data`` is a pandas object.
        target : numpy array of shape (48842,)
            Each value represents whether the person earns more than $50,000
            a year (>50K) or not (<=50K).
            If ``as_frame`` is True, ``target`` is a pandas object.
        feature_names : list of length 14
            Array of ordered feature names used in the dataset.
        DESCR : string
            Description of the UCI Adult dataset.
        categories : dict or None
            Maps each categorical feature name to a list of values, such that the
            value encoded as i is ith in the list. If ``as_frame`` is True, this is None.
        frame : pandas DataFrame
            Only present when ``as_frame`` is True. DataFrame with ``data`` and ``target``.

    (data, target) : tuple if ``return_X_y`` is True
    """

    if dataname == "adult":
        return fetch_adult(cache=cache, data_home=data_home, return_X_y=return_X_y)
    else:
        raise ValueError(f"Unsupported dataset: {dataname}")


def process_data():
    """
    [TODO] Process the loaded dataset with (1) missing value removal/imputation,
    (2) categorical feature encoding, (3) feature scaling, and (4) train-test split, etc.
    """
    pass


class DataLoader:
    """
    A class to load and process the specified dataset.

    [TODO] Provide a unified interface to load and process datasets.
    """

    def __init__(self) -> None:
        pass


# %%

if __name__ == "__main__":  # pragma: no cover
    # For local testing/debugging purposes
    data = load_data(dataname="adult")
    print(data.frame.head())
    print(data.DESCR)
    data = load_data(dataname="unknown")  # expect ValueError
