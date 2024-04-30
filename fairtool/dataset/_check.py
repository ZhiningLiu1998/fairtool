# %%

LOCAL_DEBUG = True

if not LOCAL_DEBUG:
    from ..utils._validation_params import check_type
else:  # pragma: no cover
    # For local debugging purposes
    import sys

    sys.path.append("..")
    from utils._validation_params import check_type

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


def check_X_y_s(
    X, y, s, accept_non_pandas=True, accept_non_numerical=False, return_pandas=True
):
    """Check if the input data and attributes are valid for binary classification.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data.

    y : array-like of shape (n_samples,)
        The target attribute values.

    s : array-like of shape (n_samples,)
        The sensitive attribute values.

    accept_non_pandas : bool, default=True
        Whether to accept non-pandas input data and attributes.

    accept_non_numerical : bool, default=False
        Whether to accept non-numerical target and sensitive attribute values.
        Default is False.

    return_pandas : bool, default=True
        Whether to return the input data and attributes as pandas DataFrame and Series.

    Returns
    -------
    X : pandas DataFrame
        The validated input data.

    y : pandas Series
        The validated target attribute values.

    s : pandas Series
        The validated sensitive attribute values.
    """
    check_type(accept_non_pandas, "accept_non_pandas", bool)
    check_type(accept_non_numerical, "accept_non_numerical", bool)
    check_type(return_pandas, "return_pandas", bool)
    if not isinstance(X, pd.DataFrame):
        if accept_non_pandas:
            try:
                X = pd.DataFrame(X)
            except Exception as e:
                raise ValueError(
                    f"X ({type(X)}) is not a pandas DataFrame and cannot be converted to one. "
                    f"Make sure X is an array-like of shape (n_samples, n_features)."
                )
        else:
            raise ValueError(
                f"X should be a pandas DataFrame when `accept_non_pandas` is False, got {type(X)}"
            )
    if not isinstance(y, pd.Series):
        if accept_non_pandas:
            try:
                y = pd.Series(y, name="class")
            except Exception as e:
                raise ValueError(
                    f"y ({type(y)}) is not a pandas Series and cannot be converted to one. "
                    f"Make sure y is an array-like of shape (n_samples,)."
                )
        else:
            raise ValueError(
                f"y should be a pandas Series when `accept_non_pandas` is False, got {type(y)}"
            )
    if not isinstance(s, pd.Series):
        if accept_non_pandas:
            try:
                s = pd.Series(s, name="sensitive")
            except Exception as e:
                raise ValueError(
                    f"s ({type(s)}) is not a pandas Series and cannot be converted to one. "
                    f"Make sure s is an array-like of shape (n_samples,)."
                )
        else:
            raise ValueError(
                f"s should be a pandas Series when `accept_non_pandas` is False, got {type(s)}"
            )
    assert (
        X.isna().sum().sum() == 0
    ), f"X contains {X.isna().sum().sum()} missing values, remove/impute missing values to avoid this error"
    assert (
        y.isna().sum() == 0
    ), f"y contains {y.isna().sum()} missing values, remove missing values to avoid this error"
    assert (
        s.isna().sum() == 0
    ), f"s contains {s.isna().sum()} missing values, remove missing values to avoid this error"

    if y.name is None:
        warnings.warn(
            f"`y` is a pandas Series but does not have a name, "
            f"it will be named as 'class' by default, "
            f"consider setting `y.name` to avoid this warning"
        )
        y.name = "class"
        if X.columns.contains("class"):
            raise ValueError(
                f"Failed to set `y.name` to 'class', 'class' is already in `X.columns`. "
                f"Explicitly set `y.name` to avoid this error."
            )
    if s.name is None:
        warnings.warn(
            f"`s` is a pandas Series but does not have a name, "
            f"trying to locate `s` in X and assign the column name to `s`, "
            f"consider setting `s.name` to avoid this warning"
        )
        s_col = check_s_in_X(X, s)
        if s_col == False:
            warnings.warn(
                f"`s` is not found in `X`, it will be named as 'sensitive' by default, "
            )
            s.name = "sensitive"
            if X.columns.contains("sensitive"):
                raise ValueError(
                    f"Failed to set `s.name` to 'sensitive', 'sensitive' is already in `X.columns`. "
                    f"Explicitly set `s.name` to avoid this error."
                )
        else:
            s.name = X.columns[s_col]
            warnings.warn(
                f"`s` is found in `X` at index {s_col}, assigning the column name {s.name} to `s`."
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
    if return_pandas:
        return X, y, s
    else:
        return X.values, y.values, s.values


def check_s_in_X(X, s):
    """
    Check if s is a column in X and return the index of the column if it is.
    """
    X, s = check_X_y(X, s)
    # get all columns in X that are equal to s
    candidates = [i for i in range(X.shape[1]) if np.all(X[:, i] == s)]
    if len(candidates) == 0:
        return False
    elif len(candidates) > 1:
        raise ValueError(
            f"More than one column in X is equal to s: columns {candidates} "
            f"are identical and all equal to s. Please check your data and "
            f"remove redundant columns."
        )
    else:
        return candidates[0]


def remove_s_in_X(X, s, return_pandas=False):
    """
    Remove column s from X and return the resulting array or DataFrame.
    """
    col = check_s_in_X(X, s)
    if col is False:
        raise ValueError(
            "Column s not found in X. Please check your data and make sure "
            "that s is a column in X."
        )
    if return_pandas:
        return pd.DataFrame(X).drop(X.columns[col], axis=1)
    else:
        return np.delete(X, col, axis=1)
