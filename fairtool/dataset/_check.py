import warnings

import numpy as np
import pandas as pd
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted, column_or_1d


def check_sklearn_transformer_is_fitted(transformer, error_msg=None):
    """Check if a sklearn transformer is fitted.

    Parameters
    ----------
    transformer : object
        The transformer object to check.

    error_msg : str or None, default=None
        The error message to raise if the transformer is not fitted.

    Raises
    ------
    ValueError
        If the transformer is not fitted.
    """
    try:
        check_is_fitted(transformer)
        return
    except Exception as e:
        raise ValueError(error_msg)


def check_sklearn_transformer_is_not_fitted(transformer, error_msg=None):
    """Check if a sklearn transformer is not fitted.

    Parameters
    ----------
    transformer : object
        The transformer object to check.

    error_msg : str or None, default=None
        The error message to raise if the transformer is fitted.

    Raises
    ------
    ValueError
        If the transformer is fitted.
    """
    try:
        check_is_fitted(transformer)
        raise ValueError(error_msg)
    except Exception as e:
        return


def check_target_attr(y, accept_non_numerical=False):
    """Check if the target attribute is valid for binary classification.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        The target attribute values.

    accept_non_numerical : bool, default=False
        Whether to accept non-numerical target attribute values.

    Returns
    -------
    y : array-like of shape (n_samples,)
        The validated target attribute values.

    Raises
    ------
    ValueError
        If the target attribute values contain missing values,
        are not binary, or are not numerical (if accept_non_numerical is False).
    """
    y = column_or_1d(y, dtype=None, warn=False)
    if pd.isnull(y).any():
        raise ValueError(
            f"Target values should not contain missing values, got {pd.isnull(y).sum()} missing values"
        )
    y_uniques = np.unique(y)
    if y_uniques.size != 2:
        raise ValueError(
            f"Targets should be binary, got {y_uniques.size} unique values: {y_uniques}"
        )
    if not accept_non_numerical:
        if not np.issubdtype(y.dtype, np.number):
            raise ValueError(f"Targets should be numerical, got {y.dtype} data type")
        if not set(y_uniques) == set([0, 1]):
            raise ValueError(f"Target values should be 0 and 1, got {y_uniques}")
    return y


def check_sensitive_attr(s, accept_non_numerical=False):
    """Check if the sensitive attribute is valid for binary classification.

    Parameters
    ----------
    s : array-like of shape (n_samples,)
        The sensitive attribute values.

    accept_non_numerical : bool, default=False
        Whether to accept non-numerical sensitive attribute values.

    Returns
    -------
    s : array-like of shape (n_samples,)
        The validated sensitive attribute values.

    Raises
    ------
    ValueError
        If the sensitive attribute values contain missing values,
        are not binary, or are not numerical (if accept_non_numerical is False).
    """
    s = column_or_1d(s, dtype=None, warn=False)
    if pd.isnull(s).any():
        raise ValueError(
            f"Sensitive attribute should not contain missing values, got {pd.isnull(s).sum()} missing values"
        )
    s_uniques = np.unique(s)
    if s_uniques.size != 2:
        raise ValueError(
            f"Sensitive attribute should be binary, got {s_uniques.size} unique values: {s_uniques}"
        )
    if not accept_non_numerical:
        if not np.issubdtype(s.dtype, np.number):
            raise ValueError(
                f"Sensitive attribute should be numerical, got {s.dtype} data type"
            )
        if not set(s_uniques) == set([0, 1]):
            raise ValueError(
                f"Sensitive attribute values should be 0 and 1, got {s_uniques}"
            )
    return s


def check_target_and_sensitive_attr(y, s, accept_non_numerical=False):
    """Check if the target and sensitive attributes are valid for binary classification.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        The target attribute values.

    s : array-like of shape (n_samples,)
        The sensitive attribute values.

    accept_non_numerical : bool, default=False
        Whether to accept non-numerical target and sensitive attribute values.

    Returns
    -------
    y : array-like of shape (n_samples,)
        The validated target attribute values.

    s : array-like of shape (n_samples,)
        The validated sensitive attribute values.

    Raises
    ------
    ValueError
        If the target or sensitive attribute values contain missing values,
        are not binary, or are not numerical (if accept_non_numerical is False),
        or if there is an empty subgroup detected.
    """
    # check if there is empty subgroup
    y = check_target_attr(y, accept_non_numerical=accept_non_numerical)
    s = check_sensitive_attr(s, accept_non_numerical=accept_non_numerical)
    y_uniques, s_uniques = np.unique(y), np.unique(s)
    for y_val in y_uniques:
        for s_val in s_uniques:
            n = np.sum((y == y_val) & (s == s_val))
            if n == 0:
                raise ValueError(
                    f"Empty subgroup detected: target={y_val}, sensitive={s_val}"
                )
    return y, s


def check_target_name(data, target_name, accept_non_numerical=False):
    """Check if the target attribute name is present in the DataFrame and valid for binary classification.

    Parameters
    ----------
    data : pandas DataFrame
        The DataFrame containing the dataset.

    target_name : str
        The name of the target attribute column in the DataFrame.

    accept_non_numerical : bool, default=False
        Whether to accept non-numerical target attribute values.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If `data` is not a pandas DataFrame or if `target_name` is not present in the DataFrame.

    ValueError
        If the target attribute values in the DataFrame contain missing values,
        are not binary, or are not numerical (if accept_non_numerical is False).
    """
    assert isinstance(
        data, pd.DataFrame
    ), f"`data` should be a pandas DataFrame, got {type(data)} instead"
    assert (
        target_name in data.columns
    ), f"`target_name` should be present in the DataFrame, got {target_name}"
    check_target_attr(data[target_name], accept_non_numerical=accept_non_numerical)
    return


def check_sensitive_name(data, sensitive_name, accept_non_numerical=False):
    """Check if the sensitive attribute name is present in the DataFrame and valid for binary classification.

    Parameters
    ----------
    data : pandas DataFrame
        The DataFrame containing the dataset.

    sensitive_name : str
        The name of the sensitive attribute column in the DataFrame.

    accept_non_numerical : bool, default=False
        Whether to accept non-numerical sensitive attribute values.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If `data` is not a pandas DataFrame or if `sensitive_name` is not present in the DataFrame.

    ValueError
        If the sensitive attribute values in the DataFrame contain missing values,
        are not binary, or are not numerical (if accept_non_numerical is False).
    """
    assert isinstance(
        data, pd.DataFrame
    ), f"`data` should be a pandas DataFrame, got {type(data)} instead"
    assert (
        sensitive_name in data.columns
    ), f"`sensitive_name` should be present in the DataFrame, got {sensitive_name}"
    check_sensitive_attr(
        data[sensitive_name], accept_non_numerical=accept_non_numerical
    )
    return


def check_feature_names(data, feature_names, accept_non_numerical=False):
    """Check if the feature names are present in the DataFrame and valid for binary classification.

    Parameters
    ----------
    data : pandas DataFrame
        The DataFrame containing the dataset.

    feature_names : list of str
        The list of feature names.

    accept_non_numerical : bool, default=False
        Whether to accept non-numerical feature values.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If `data` is not a pandas DataFrame, if `feature_names` is not a list,
        or if any feature name is not present in the DataFrame.

    ValueError
        If any feature values in the DataFrame contain missing values,
        are not numerical (if accept_non_numerical is False), or
        if any feature name contains '=' or '_'.
    """
    assert isinstance(
        data, pd.DataFrame
    ), f"`data` should be a pandas DataFrame, got {type(data)} instead"
    assert (
        column_or_1d(feature_names, dtype=None, warn=False) is not None
    ), f"`feature_names` should be a list of feature names, got {feature_names} instead"
    assert all(
        [col in data.columns for col in feature_names]
    ), f"All feature names should be present in the DataFrame, got {feature_names}"
    if not accept_non_numerical:
        for col in feature_names:
            if not np.issubdtype(data[col].dtype, np.number):
                raise ValueError(
                    f"Feature '{col}' should be numerical, got {data[col].dtype} data type"
                )
    # feature names should not contain '=' or '_'
    for col in feature_names:
        if "=" in col or "_" in col:
            raise ValueError(
                f"Feature name should not contain '=' or '_', they will be used "
                f"for naming the one-hot encoded features, got '{col}'"
            )
    return


def check_target_and_sensitive_names(
    data, target_name, sensitive_name, accept_non_numerical=False
):
    """Check if the target and sensitive attribute names are present in the
    DataFrame and valid for binary classification.

    Parameters
    ----------
    data : pandas DataFrame
        The DataFrame containing the dataset.

    target_name : str
        The name of the target attribute column in the DataFrame.

    sensitive_name : str
        The name of the sensitive attribute column in the DataFrame.

    accept_non_numerical : bool, default=False
        Whether to accept non-numerical target and sensitive attribute values.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the target or sensitive attribute names are not present in the DataFrame,
        or if any attribute values in the DataFrame are invalid.
    """
    check_target_name(data, target_name, accept_non_numerical=accept_non_numerical)
    check_sensitive_name(
        data, sensitive_name, accept_non_numerical=accept_non_numerical
    )
    check_target_and_sensitive_attr(
        data[target_name],
        data[sensitive_name],
        accept_non_numerical=accept_non_numerical,
    )
    return


def check_feature_target_sensitive_names(
    data, feature_names, target_name, sensitive_name, accept_non_numerical=False
):
    """Check if the feature, target, and sensitive attribute names are present
    in the DataFrame and valid for binary classification.

    Parameters
    ----------
    data : pandas DataFrame
        The DataFrame containing the dataset.

    feature_names : list of str
        The list of feature names.

    target_name : str
        The name of the target attribute column in the DataFrame.

    sensitive_name : str
        The name of the sensitive attribute column in the DataFrame.

    accept_non_numerical : bool, default=False
        Whether to accept non-numerical feature, target, and sensitive attribute values.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any attribute names are not present in the DataFrame, if any attribute
        values in the DataFrame are invalid, or if the target attribute is included
        in the feature names.
    """
    check_feature_names(data, feature_names, accept_non_numerical=accept_non_numerical)
    check_target_and_sensitive_names(
        data, target_name, sensitive_name, accept_non_numerical=accept_non_numerical
    )
    if target_name in feature_names:
        raise ValueError(
            f"Target attribute '{target_name}' should not be included in the feature names"
        )
    return


def check_X_y_s(X, y, s, accept_non_numerical=False):
    """Check if the input data and attributes are valid for binary classification.

    Parameters
    ----------
    X : pandas DataFrame
        The input data.

    y : pandas Series
        The target attribute values.

    s : pandas Series
        The sensitive attribute values.

    accept_non_numerical : bool, default=False
        Whether to accept non-numerical target and sensitive attribute values.
        Default is False.

    Returns
    -------
    X : pandas DataFrame
        The validated input data.

    y : pandas Series
        The validated target attribute values.

    s : pandas Series
        The validated sensitive attribute values.
    """
    assert isinstance(X, pd.DataFrame), f"X must be a pandas DataFrame, got {type(X)}"
    assert isinstance(y, pd.Series), f"y must be a pandas Series, got {type(y)}"
    assert isinstance(s, pd.Series), f"s must be a pandas Series, got {type(s)}"
    if y.name is None:
        warnings.warn(
            f"y is a pandas Series but does not have a name, "
            f"it will be named as 'class' by default, "
            f"consider setting y.name to avoid this warning"
        )
        y.name = "class"
    if s.name is None:
        raise ValueError(
            f"Sensitive attribute `s` does not have a name. "
            f"it must have a name in order to check whether it is "
            f"included in X or conflicting with y. Explicitly set "
            f"s.name to avoid this error."
        )
    # check validitiy of target and sensitive attributes and empty subgroups
    check_target_and_sensitive_attr(y, s, accept_non_numerical=accept_non_numerical)
    check_X_y(X, y)
    check_X_y(X, s)
    # y.name should not be in X.columns
    assert (
        y.name not in X.columns
    ), f"y.name {y.name} is in X.columns, remove y from X to avoid this error"
    # y.name should not be equal to s.name
    assert (
        s.name != y.name
    ), f"s.name {s.name} is equal to y.name, change the name of s or y to avoid this error"
    return X, y, s
