import pandas as pd
import numpy as np
import sklearn
from sklearn.impute._base import _BaseImputer
from pandas.api.types import is_numeric_dtype, CategoricalDtype


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
