# %%

LOCAL_DEBUG = True

if not LOCAL_DEBUG:
    from ._check import check_target_and_sensitive_attr, check_X_y_s
    from ..utils._validation_params import check_type
else:  # pragma: no cover
    # For local debugging purposes
    import sys

    sys.path.append("..")
    from dataset._check import check_target_and_sensitive_attr, check_X_y_s
    from utils._validation_params import check_type

import warnings

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype, is_numeric_dtype
from sklearn.impute._base import _BaseImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.validation import check_random_state

# %%


def train_val_test_split(
    X,
    y,
    s,
    *,
    test_ratio,
    val_ratio=0,
    stratify_subgroup=True,
    stratify=None,
    return_indices=False,
    random_state=None,
):
    """
    Split the data into train, validation, and test sets.

    Parameters
    ----------
    X : array-like or dataframe, shape (n_samples, n_features)
        The input features.

    y : array-like, shape (n_samples,)
        The target variable.

    s : array-like, shape (n_samples,)
        The sensitive attribute.

    test_ratio : float
        The ratio of the test set size to the total dataset size.
        Must be in the range [0, 1).

    val_ratio : float, optional (default=0)
        The ratio of the validation set size to the training set size.
        Must be in the range [0, 1).

    stratify_subgroup : bool, optional (default=True)
        Whether to perform subgroup stratification based on the sensitive attribute and target variable.
        If True, ensures balanced subgroup distribution across splits.

    stratify : array-like, shape (n_samples,), optional (default=None)
        An array indicating the groups for subgroup stratification.
        If provided, overrides automatic subgroup stratification based on y and s.

    return_indices : bool, optional (default=False)
        Whether to return the indices of train, validation, and test sets instead of the actual splits.

    random_state : int or RandomState instance, optional (default=None)
        Controls the randomness of the split.

    Returns
    -------
    splits : list, length=9 if `val_ratio` > 0, else length=6
        A list containing the train, validation (if any), and test splits of inputs.
        If `val_ratio`>0, return [X_train, X_val, X_test, y_train, y_val, y_test, s_train, s_val, s_test].
        If `val_ratio`=0, return [X_train, X_test, y_train, y_test, s_train, s_test].

    indices : list, length=3 if `val_ratio` > 0, else length=2
        A list containing the indices of train, validation (if any), and test sets.
        If `val_ratio`>0, return [train_index, val_index, test_index].
        If `val_ratio`=0, return [train_index, test_index].

    Raises
    ------
    AssertionError
        If test_ratio or val_ratio is not in the range [0, 1).
        If the sum of val_ratio and test_ratio is not less than 1.

    Warnings
    --------
    UserWarning
        If stratify_subgroup is set to False, which may lead to imbalanced subgroup distribution in the splits.
        If a custom stratify array is passed, which overrides automatic subgroup stratification.
    """
    check_X_y_s(X, y, s, accept_non_numerical=False)
    check_type(test_ratio, "test_ratio", (int, float))
    check_type(val_ratio, "val_ratio", (int, float))
    assert (
        1 > test_ratio >= 0
    ), f"test_ratio must be in the range [0, 1), got {test_ratio}."
    assert (
        1 > val_ratio >= 0
    ), f"val_ratio must be in the range [0, 1), got {val_ratio}."
    assert (
        val_ratio + test_ratio < 1
    ), f"The sum of val_ratio (got {val_ratio}) and test_ratio (got {test_ratio}) must be less than 1."
    check_type(stratify_subgroup, "stratify_subgroup", bool)
    random_state = check_random_state(random_state)

    if stratify_subgroup != True:
        warnings.warn(
            f"The stratify_subgroup parameter is set to False. This may lead "
            f"to imbalanced subgroup (determined by y and s) distribution in the splits, "
            f"some splits may have no samples from one or more subgroups. "
            f"We recommend setting stratify_subgroup=True to ensure balanced subgroup distribution."
        )

    if stratify is not None:
        warnings.warn(
            f"You have passed a custom stratify array. This will override the automatic "
            f"subgroup stratification based on the target (y) and sensitive attribute (s). "
            f"This may lead to imbalanced subgroup distribution in the splits, and some "
            f"splits may have no samples from one or more subgroups. Make sure you are "
            f"aware of the consequences of this action."
        )

    # compute stratify array based on y and s
    if stratify is None and stratify_subgroup == True:
        stratify = np.zeros_like(y)
        stratify = s * 2 + y

    sss_test = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_ratio,
        random_state=random_state,
    )
    train_index, test_index = next(sss_test.split(X, stratify))

    if val_ratio > 0:
        # then train/val split
        new_stratify = stratify[train_index].values
        sss_val = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_ratio / (1 - test_ratio),
            random_state=random_state,
        )
        train_index_, val_index_ = next(
            sss_val.split(np.zeros_like(new_stratify), new_stratify)
        )
        # translate to true index
        true_index = train_index.copy()
        train_index, val_index = true_index[train_index_], true_index[val_index_]
        indices = [train_index, val_index, test_index]
    else:
        indices = [train_index, test_index]

    if return_indices:
        return indices
    else:
        splits = []
        for v in [X, y, s]:
            for idx in indices:
                splits.append(v.loc[v.index[idx]])
        return splits


def process_missing_values(data, how="drop", imputer=None):
    """
    Handle missing values in the DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to process.

    how : str, default='drop'
        How to handle missing values. Possible values are 'drop' and 'impute'.
        if 'drop', rows with missing values will be dropped.
        if 'impute', missing values will be imputed using the provided imputer.

    imputer : object, default=None
        The sklearn feature imputer to use if how='impute'.

    Returns
    -------
    df : pd.DataFrame
        The DataFrame after processing missing values.
    """
    assert isinstance(
        data, pd.DataFrame
    ), f"`data` should be a pandas DataFrame, got {type(data)} instead"
    assert how in [
        "drop",
        "impute",
    ], f"`how` should be either 'drop' or 'impute', got {how} instead"
    if how == "impute":
        assert isinstance(
            imputer, _BaseImputer
        ), f"`imputer` should be an sklearn imputer, got {type(imputer)} instead"
        df = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    elif how == "drop":
        df = data.dropna()
    return df


def set_feature_dtypes(
    df: pd.DataFrame, categorical_features: list = None, verbose: bool = False
):
    """
    Set the feature data types of the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to set the feature data types.

    categorical_features : list, default=None
        The list of categorical features. If None, categorical features will
        be inferred from the DataFrame.

    verbose : bool, default=False
        Whether to print additional information.

    Returns
    -------

    """
    assert isinstance(
        df, pd.DataFrame
    ), f"`df` should be a pandas DataFrame, got {type(df)} instead"
    if categorical_features is not None:
        assert isinstance(
            categorical_features, list
        ), f"`categorical_features` should be a list, got {type(categorical_features)} instead"
        assert set(categorical_features).issubset(set(df.columns)), (
            f"All features in `categorical_features` should be present in the DataFrame, "
            f"got invalid feature name(s) in `categorical_features`: "
            f"{set(categorical_features).difference(set(df.columns))}."
        )

    # get target feature types
    if categorical_features is None:  # if not specified, parse from the input data
        feat_dtypes = parse_feature_dtypes(df, verbose=verbose)
    else:
        feat_dtypes = {
            col: "categorical" if col in categorical_features else "numerical"
            for col in df.columns
        }

    # convert the feature data types
    for column, dtype in feat_dtypes.items():
        if dtype == "numerical":
            try:
                df.loc[:, column] = pd.to_numeric(df[column])
            except Exception as e:
                raise ValueError(
                    f"Column {column} cannnot be set to numerical. Error: {e}"
                )
        elif dtype == "categorical":
            try:
                df.loc[:, column] = df[column].astype("category")
            except Exception as e:
                raise ValueError(
                    f"Column {column} cannnot be set to categorical. Error: {e}"
                )
    return df


def parse_feature_dtypes(
    df: pd.DataFrame,
    feature_columns: list = None,
    dtype_as_key: bool = False,
    verbose: bool = False,
):
    """
    Parse the feature data types of the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to parse the feature data types.

    feature_columns : list, default=None
        The list of feature columns to parse. If None, all columns in the DataFrame
        will be parsed.

    verbose : bool, default=False
        Whether to print runtime information.a

    Returns
    -------
    feat_dtypes : dict
        A dictionary with keys as the feature names and values as the feature data types.
    """
    assert isinstance(
        df, pd.DataFrame
    ), f"`df` should be a pandas DataFrame, got {type(df)} instead"
    if feature_columns is not None:
        assert isinstance(
            feature_columns, list
        ), f"`feature_columns` should be a list, got {type(feature_columns)} instead"
        assert set(feature_columns).issubset(set(df.columns)), (
            f"All features in `feature_columns` should be present in the DataFrame, "
            f"got invalid feature name(s) in `feature_columns`: "
            f"{set(feature_columns).difference(set(df.columns))}."
        )
    assert isinstance(
        verbose, bool
    ), f"`verbose` should be a boolean, got {type(verbose)} instead"

    if verbose:
        print(
            f"Parsing feature data types (numerical/categorical) from the DataFrame ..."
        )

    feat_dtypes = {}
    columns = df.columns if feature_columns is None else feature_columns
    for column in columns:
        dtype = df[column].dtype
        if is_numeric_dtype(dtype):
            feat_dtypes[column] = "numerical"
        elif isinstance(dtype, CategoricalDtype):
            feat_dtypes[column] = "categorical"
        else:
            if verbose:
                print(
                    f"Column '{column}' has dtype `{dtype}`. Trying to converting to categorical."
                )
            try:
                df[column] = df[column].astype("category")
                feat_dtypes[column] = "categorical"
            except Exception as e:
                raise ValueError(
                    f"Column {column} has dtype {dtype}. Cannot convert to categorical. Error: {e}"
                )

    if dtype_as_key:
        feat_dtypes = dict_inverse_collate(feat_dtypes)

    return feat_dtypes


def parse_detailed_feature_dtypes(
    df: pd.DataFrame,
    feature_columns: list = None,
    dtype_as_key: bool = False,
    verbose: bool = False,
):
    """
    Parse detailed feature data types for each column in the DataFrame.
    """
    assert isinstance(
        df, pd.DataFrame
    ), f"`df` should be a pandas DataFrame, got {type(df)} instead"
    if feature_columns is not None:
        assert isinstance(
            feature_columns, list
        ), f"`feature_columns` should be a list, got {type(feature_columns)} instead"
        assert set(feature_columns).issubset(set(df.columns)), (
            f"All features in `feature_columns` should be present in the DataFrame, "
            f"got invalid feature name(s) in `feature_columns`: "
            f"{set(feature_columns).difference(set(df.columns))}."
        )
    assert isinstance(
        verbose, bool
    ), f"`verbose` should be a boolean, got {type(verbose)} instead"
    assert isinstance(
        dtype_as_key, bool
    ), f"`dtype_as_key` should be a boolean, got {type(dtype_as_key)} instead"

    if verbose:
        print(
            f"Parsing detailed feature data types (numerical/binary-/multi-categorical) from the DataFrame ..."
        )

    feat_dtypes = {}
    columns = df.columns if feature_columns is None else feature_columns
    for column in columns:
        dtype = df[column].dtype
        if is_numeric_dtype(dtype):  # numerical
            feat_dtypes[column] = "numerical"
            continue
        else:  # try to convert to categorical
            if not isinstance(dtype, CategoricalDtype):
                if verbose:
                    print(
                        f"Column '{column}' has dtype `{dtype}`. Trying to converting to categorical."
                    )
                try:
                    df[column] = df[column].astype("category")
                except Exception as e:
                    raise ValueError(
                        f"Column {column} has dtype {dtype}. Cannot convert to categorical. Error: {e}"
                    )
            # check if binary or multi-categorical
            n_categories = len(df[column].cat.categories)
            if n_categories == 2:
                feat_dtypes[column] = "categorical-binary"
            elif n_categories > 2:
                feat_dtypes[column] = "categorical-multi"
            else:
                raise ValueError(
                    f"Column {column} should have >=2 categories, got {n_categories} instead."
                )

    if dtype_as_key:
        feat_dtypes = dict_inverse_collate(feat_dtypes)

    return feat_dtypes


def dict_inverse_collate(d):
    """
    Invert a dictionary of lists to a collated dictionary.
    """
    inv = {}
    for k, v in d.items():
        if v not in inv:
            inv[v] = []
        inv[v].append(k)
    return inv


def get_subgroup_population(y, s, return_pandas=True):
    """
    Get the population of each subgroup in the DataFrame.
    """
    check_target_and_sensitive_attr(y, s, accept_non_numerical=False)
    p = np.zeros((2, 2), dtype=int)
    for label in [0, 1]:
        for mem in [0, 1]:
            p[label, mem] = sum((y == label) & (s == mem))
    if return_pandas:
        p = pd.DataFrame(
            p, columns=pd.Series([0, 1], name="s"), index=pd.Series([0, 1], name="y")
        )
    return p


def get_group_stats(y, s):
    """
    Get the sub-group statistics for the target and sensitive attributes.
    """
    check_target_and_sensitive_attr(y, s, accept_non_numerical=False)
    p = get_subgroup_population(y, s, return_pandas=True)
    grp_stats = {
        "y_group_size": p.sum(axis=1).to_dict(),
        "y_group_ratio": (p.sum(axis=1) / p.sum().sum()).to_dict(),
        "s_group_size": p.sum(axis=0).to_dict(),
        "s_group_ratio": (p.sum(axis=0) / p.sum().sum()).to_dict(),
        "y_s_group_ratio": {
            y_label: {
                s_label: p.loc[y_label, s_label] / p.loc[y_label].sum()
                for s_label in [0, 1]
            }
            for y_label in [0, 1]
        },
        "s_y_group_ratio": {
            s_label: {
                y_label: p.loc[y_label, s_label] / p.loc[:, s_label].sum()
                for y_label in [0, 1]
            }
            for s_label in [0, 1]
        },
        "y_group_pri_ratio": {
            y_label: p.loc[y_label, 1] / p.loc[y_label].sum() for y_label in [0, 1]
        },
        "s_group_pos_ratio": {
            s_label: p.loc[1, s_label] / p.loc[:, s_label].sum() for s_label in [0, 1]
        },
    }
    return grp_stats


def dict_values_to_percentage(d, decimals=2):
    """
    Convert the values in the dictionary to percentage.
    """
    d_new = {}
    for k, v in d.items():
        d_new[k] = f"{v:.{decimals}%}"
    return d_new


# %%
