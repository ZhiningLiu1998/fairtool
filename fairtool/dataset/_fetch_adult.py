# Copyright (c) fairtool contributors.
# Distributed under the terms of the MIT License.
# Initially adapted from fairlearn (https://github.com/fairlearn/fairlearn)

"""
Utilities for fetching the UCI Adult dataset.
"""

# %%
LOCAL_DEBUG = True

if not LOCAL_DEBUG:
    from ._base import _get_download_data_home
else:  # pragma: no cover
    # For local debugging purposes
    import sys

    sys.path.append("..")
    from dataset._base import _get_download_data_home

from sklearn.datasets import fetch_openml


def fetch_adult(*, cache=True, data_home=None, as_frame=True, return_X_y=False):
    """Load the UCI Adult dataset (binary classification).

    Read more in the :ref:`User Guide <boston_housing_data>`.

    Download it if necessary.

    ==============   ====================
    Samples total                   48842
    Dimensionality                     14
    Features         numeric, categorical
    Classes                             2
    ==============   ====================

    Source:

    - UCI Repository :footcite:`kohavi1996adult`
    - Paper: Kohavi and Becker :footcite:`kohavi1996scaling`

    Prediction task is to determine whether a person makes over $50,000 a
    year.

    Parameters
    ----------
    cache : bool, default=True
        Whether to cache downloaded datasets using joblib.

    data_home : str, default=None
        Specify another download and cache folder for the datasets.
        By default, all fairlearn data is stored in '~/.fairlearn-data'
        subfolders.

    as_frame : bool, default=True
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric, string or categorical). The target is
        a pandas DataFrame or Series depending on the number of target_columns.
        The Bunch will contain a ``frame`` attribute with the target and the
        data. If ``return_X_y`` is True, then ``(data, target)`` will be pandas
        DataFrames or Series as describe above.

    return_X_y : bool, default=False
        If True, returns ``(data.data, data.target)`` instead of a Bunch
        object.

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

    Notes
    ----------
    This function is a wrapper of the :func:`sklearn.datasets.fetch_openml`.

    """
    if not data_home:
        data_home = _get_download_data_home()

    # For data_home see
    # https://github.com/scikit-learn/scikit-learn/issues/27447
    return fetch_openml(
        data_id=1590,
        data_home=str(data_home),
        cache=cache,
        as_frame=as_frame,
        return_X_y=return_X_y,
        parser="auto",
    )


# %%

if __name__ == "__main__":  # pragma: no cover
    # For local debugging purposes
    data = fetch_adult()
    print(data.frame.head())
    print(data.DESCR)

# %%
