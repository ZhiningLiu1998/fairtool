"""
This module implements the FairDataTransformer class.

TODO: 
- Implement .fit and .transform methods
"""

# %%
LOCAL_DEBUG = True

if not LOCAL_DEBUG:
    from ._check import *
    from .utils import *
else:  # pragma: no cover
    # For local debugging purposes
    import sys

    sys.path.append("..")
    from dataset._check import *
    from dataset.utils import *

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


class FairDataTransformer:
    """A class for fair data transformation.

    This class validates and (inverse) transforms the raw data into a format
    that can be used by the fair learning algorithms. The transformation includes
    encoding the target and sensitive attributes, scaling the numerical features, and
    encoding the binary and multi-categorical features. It also provides methods
    to inverse transform the transformed data back to the original format to support
    data-based unfairness explanation and bias interpretation.

    Parameters
    ----------
    feature_num_scaler : object, default=None
        The numerical feature scaler transformer. If None, a MinMaxScaler
        instance will be used with feature_range=(0, 1).

    feature_cat_bin_encoder : object, default=None
        The binary categorical feature encoder. If None, an OrdinalEncoder
        instance will be used.

    feature_cat_multi_encoder : object, default=None
        The multi-categorical feature encoder. If None, a OneHotEncoder
        instance will be used.

    Attributes
    ----------
    feature_num_scaler : object
        The fitted numerical feature scaler.

    feature_cat_bin_encoder : object
        The fitted binary categorical feature encoder.

    feature_cat_multi_encoder : object
        The fitted multi-categorical feature encoder.

    target_encoder : object
        The fitted target attribute encoder.

    sensitive_encoder : object
        The fitted sensitive attribute encoder.

    _fitted : bool
        Whether the transformer is fitted.
    """

    def __init__(
        self,
        feature_num_scaler=None,
        feature_cat_bin_encoder=None,
        feature_cat_multi_encoder=None,
    ):
        """Initialize FairDataTransformer."""
        if feature_num_scaler is None:
            self.feature_num_scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
        else:
            scaler_classes = (
                sklearn.base.BaseEstimator,
                sklearn.base.TransformerMixin,
                sklearn.base.OneToOneFeatureMixin,
            )
            assert isinstance(feature_num_scaler, scaler_classes), (
                f"Invalid `feature_num_scaler`={feature_num_scaler}, "
                f"it should be a valid sklearn transformer instance "
                f"inheriting from {scaler_classes}"
            )
            self.feature_num_scaler = feature_num_scaler

        if feature_cat_bin_encoder is None:
            self.feature_cat_bin_encoder = OrdinalEncoder()
        else:
            encoder_class = sklearn.preprocessing._encoders._BaseEncoder
            assert isinstance(feature_cat_bin_encoder, encoder_class), (
                f"Invalid `feature_cat_bin_encoder`={feature_cat_bin_encoder}, "
                f"it should be a valid sklearn encoder instance inheriting from {encoder_class}"
            )
            self.feature_cat_bin_encoder = feature_cat_bin_encoder

        if feature_cat_multi_encoder is None:
            self.feature_cat_multi_encoder = OneHotEncoder()
        else:
            encoder_class = sklearn.preprocessing._encoders._BaseEncoder
            assert isinstance(feature_cat_multi_encoder, encoder_class), (
                f"Invalid `feature_cat_multi_encoder`={feature_cat_multi_encoder}, "
                f"it should be a valid sklearn encoder instance inheriting from {encoder_class}"
            )
            self.feature_cat_multi_encoder = feature_cat_multi_encoder

        self.target_encoder = OrdinalEncoder()
        self.sensitive_encoder = OrdinalEncoder()
        self._fitted = False

        return

    def fit_transform_X_y_s_from_dataframe(
        self,
        data,
        target_name,
        sensitive_name,
        feature_names=None,
        categorical_features=None,
        exclude_sensitive_feature=False,
        target_pos_value=None,
        sensitive_pos_value=None,
        verbose=False,
    ):
        """Fit the transformer and transform the dataset.

        Parameters
        ----------
        data : pandas DataFrame
            The input dataset.

        target_name : str
            The name of the target attribute column.

        sensitive_name : str
            The name of the sensitive attribute column.

        feature_names : list of str or None, default=None
            The list of feature column names. If not specified, all columns
            except the target will be used as features. If `exclude_sensitive_feature`
            is True, the sensitive feature will be also be excluded.

        categorical_features : list of str or None, default=None
            The list of categorical feature names. If None, categorical
            features will be inferred from the DataFrame.

        exclude_sensitive_feature : bool, default=False
            Whether to exclude the sensitive feature. Default is False.
            If True, the sensitive feature will be excluded from the
            transformed feature matrix. Ignored if `feature_names` is specified.

        target_pos_value : object or None, default=None
            The positive value of the target attribute. If not specified,
            the minority class will be used as the positive value.

        sensitive_pos_value : object or None, default=None
            The positive value of the sensitive attribute. If not specified,
            the advantageous group (with high positive label ratio) will be
            used as the positive value.

        verbose : bool, default=False
            Whether to print verbose output. Default is False.

        Returns
        -------
        X : pandas DataFrame
            The transformed feature matrix.

        y : pandas Series
            The transformed target attribute.

        s : pandas Series
            The transformed sensitive attribute.
        """
        if feature_names is None:
            feature_names = data.columns.tolist()
            feature_names.remove(target_name)
            if exclude_sensitive_feature:
                feature_names.remove(sensitive_name)
        check_feature_target_sensitive_names(
            data, feature_names, target_name, sensitive_name, accept_non_numerical=True
        )
        X, y, s = data[feature_names], data[target_name], data[sensitive_name]
        # set the feature dtypes (automatically determined if categorical_features=None)
        X = set_feature_dtypes(
            X, categorical_features=categorical_features, verbose=verbose
        )

        # parse the feature dtypes, keys are 'numerical', 'categorical-binary', 'categorical-multi'
        dtype_features = parse_detailed_feature_dtypes(
            X, feature_columns=feature_names, dtype_as_key=True, verbose=False
        )
        # rearrange raw features in the order of numerical, binary categorical, and multi-categorical
        feature_names = (
            dtype_features["numerical"]
            + dtype_features["categorical-binary"]
            + dtype_features["categorical-multi"]
        )
        self.feature_names = feature_names
        self.target_name = target_name
        self.sensitive_name = sensitive_name
        # store the data with rearranged columns
        self.data_raw = data[
            [f for f in feature_names if f != sensitive_name]
            + [sensitive_name, target_name]
        ]

        # transform the target and sensitive attribute first
        y_ = self._fit_transform_target_attr(y, pos_value=target_pos_value)
        s_ = self._fit_transform_sensitive_attr(s, y=y_, pos_value=sensitive_pos_value)

        # transform the features
        X_num_ = self._fit_transform_numerical_features(X[dtype_features["numerical"]])
        X_bin_ = self._fit_transform_categorical_binary_features(
            X[dtype_features["categorical-binary"]], sensitive_name
        )
        X_multi_ = self._fit_transform_categorical_multi_features(
            X[dtype_features["categorical-multi"]]
        )
        X_ = pd.concat([X_num_, X_bin_, X_multi_], axis=1)

        # store the input/output feature names and their types
        self.feature_names_out_ = (
            X_num_.columns.tolist()
            + X_bin_.columns.tolist()
            + X_multi_.columns.tolist()
        )
        self.feature_names_out_type_dict_ = {
            "numerical": X_num_.columns.tolist(),
            "categorical-binary": X_bin_.columns.tolist(),
            "categorical-multi": X_multi_.columns.tolist(),
        }
        self.feature_names_in_ = feature_names
        self.feature_names_in_type_dict_ = dtype_features
        self._fitted = True

        self.X_out_ = X_
        self.y_out_ = y_
        self.s_out_ = s_

        return X_, y_, s_

    def inverse_transform_X(self, X):
        """Inverse transform the features.

        Parameters
        ----------
        X : pandas DataFrame
            The transformed feature matrix.

        Returns
        -------
        pandas DataFrame
            The inverse transformed feature matrix.
        """
        assert (
            self._fitted
        ), "The transformer is not fitted, call `fit_transform_X_y_s_from_dataframe` before calling `inverse_transform_X`"
        in_feats = self.feature_names_in_type_dict_
        out_feats = self.feature_names_out_type_dict_

        X_num_ = self._inverse_transform_numerical_features(X[out_feats["numerical"]])
        X_bin_ = self._inverse_transform_categorical_binary_features(
            X[out_feats["categorical-binary"]]
        )
        X_multi_ = self._inverse_transform_categorical_multi_features(
            X[out_feats["categorical-multi"]]
        )
        X_num_ = pd.DataFrame(X_num_, columns=in_feats["numerical"])
        X_bin_ = pd.DataFrame(X_bin_, columns=in_feats["categorical-binary"])
        X_multi_ = pd.DataFrame(X_multi_, columns=in_feats["categorical-multi"])
        return pd.concat([X_num_, X_bin_, X_multi_], axis=1)

    def inverse_transform_y(self, y):
        """Inverse transform the target attribute.

        Parameters
        ----------
        y : pandas Series
            The transformed target attribute.

        Returns
        -------
        pandas Series
            The inverse transformed target attribute.
        """
        return self._inverse_transform_target_attr(y)

    def inverse_transform_s(self, s):
        """Inverse transform the sensitive attribute.

        Parameters
        ----------
        s : pandas Series
            The transformed sensitive attribute.

        Returns
        -------
        pandas Series
            The inverse transformed sensitive attribute.
        """
        return self._inverse_transform_sensitive_attr(s)

    @property
    def X_in_(self):
        """Get the input feature matrix.

        Returns
        -------
        pandas DataFrame
            The input feature matrix.
        """
        assert (
            self._fitted
        ), "The transformer is not fitted, call `fit_transform_X_y_s_from_dataframe` before accessing `X_in_`"
        return self.data_raw[self.feature_names_in_]

    @property
    def y_in_(self):
        """Get the input target attribute.

        Returns
        -------
        pandas Series
            The input target attribute.
        """
        assert (
            self._fitted
        ), "The transformer is not fitted, call `fit_transform_X_y_s_from_dataframe` before accessing `y_in_`"
        return self.data_raw[self.target_name]

    @property
    def s_in_(self):
        """Get the input sensitive attribute.

        Returns
        -------
        pandas Series
            The input sensitive attribute.
        """
        assert (
            self._fitted
        ), "The transformer is not fitted, call `fit_transform_X_y_s_from_dataframe` before accessing `s_in_`"
        return self.data_raw[self.sensitive_name]

    def _fit_transform_numerical_features(self, X_num):
        """Fit and transform the numerical features.

        Parameters
        ----------
        X_num : pandas DataFrame
            The numerical feature matrix.

        Returns
        -------
        pandas DataFrame
            The transformed numerical feature matrix.

        Raises
        ------
        RuntimeError
            If failed to transform numerical features.
        """
        check_sklearn_transformer_is_not_fitted(
            self.feature_num_scaler,
            "A numerical feature scaler is already fitted, create a new instance of DataTransformer to transform a new dataset",
        )
        try:
            X_num_ = self.feature_num_scaler.fit_transform(X_num)
        except Exception as e:
            raise RuntimeError(
                f"Failed to transform numerical feature with the provided feature_num_scaler={self.feature_num_scaler}, make sure the scaler you provide is valid: {e}"
            )
        return pd.DataFrame(X_num_, columns=X_num.columns, index=X_num.index)

    def _inverse_transform_numerical_features(self, X_num):
        """Inverse transform numerical features.

        Parameters
        ----------
        X_num : pandas DataFrame
            The transformed numerical feature matrix.

        Returns
        -------
        pandas DataFrame
            The inverse transformed numerical feature matrix.

        Raises
        ------
        RuntimeError
            If failed to inverse transform numerical features.

        """
        check_sklearn_transformer_is_fitted(
            self.feature_num_scaler,
            "Failed to inverse transform numerical feature, the numerical feature scaler is not fitted yet",
        )
        try:
            return self.feature_num_scaler.inverse_transform(X_num)
        except Exception as e:
            raise RuntimeError(
                f"Failed to inverse transform numerical feature with the provided data, make sure the data you provide is valid: {e}"
            )

    def _fit_transform_categorical_binary_features(self, X_bin, sensitive_name):
        """Fit and transform binary categorical features.

        Parameters
        ----------
        X_bin : pandas DataFrame
            The binary categorical feature matrix.

        sensitive_name : str
            The name of the sensitive attribute column. The encoding rule for
            the sensitive attribute will be used in a consistent manner if the
            sensitive attribute is included in the features.

        Returns
        -------
        pandas DataFrame
            The transformed binary categorical feature matrix.

        Raises
        ------
        RuntimeError
            If failed to transform binary categorical features.

        """
        check_sklearn_transformer_is_not_fitted(
            self.feature_cat_bin_encoder,
            "Failed to transform binary categorical feature, the binary categorical encoder is already fitted. "
            + "Create a new instance of DataTransformer to transform a new dataset.",
        )

        categories = []
        for col in X_bin.columns:
            if (
                col == sensitive_name
            ):  # for the sensitive attribute, use the same encoding rule
                check_sklearn_transformer_is_fitted(
                    self.sensitive_encoder,
                    "The sensitive attribute is included in the features, but the sensitive_encoder is not fitted yet. "
                    + "Check the order of the transformation steps to guarantee consistent encoding for the sensitive feature",
                )
                categories.append(self.sensitive_encoder.categories_[0])
            else:
                categories.append(X_bin[col].unique())

        try:
            self.feature_cat_bin_encoder.set_params(categories=categories)
            X_bin_ = self.feature_cat_bin_encoder.fit_transform(X_bin)
        except Exception as e:
            raise RuntimeError(
                f"Failed to transform binary categorical feature with the provided encoder "
                f"({self.feature_cat_bin_encoder}), make sure the encoder you provide is valid: {e}"
            )
        return pd.DataFrame(X_bin_, columns=X_bin.columns, index=X_bin.index)

    def _inverse_transform_categorical_binary_features(self, X_bin):
        """Inverse transform binary categorical features.

        Parameters
        ----------
        X_bin : pandas DataFrame
            The transformed binary categorical feature matrix.

        Returns
        -------
        pandas DataFrame
            The inverse transformed binary categorical feature matrix.

        Raises
        ------
        RuntimeError
            If failed to inverse transform binary categorical features.
        """
        check_sklearn_transformer_is_fitted(
            self.feature_cat_bin_encoder,
            "Failed to inverse transform binary categorical feature, the binary categorical encoder is not fitted yet",
        )
        try:
            return self.feature_cat_bin_encoder.inverse_transform(X_bin)
        except Exception as e:
            raise RuntimeError(
                f"Failed to inverse transform binary categorical feature with the provided "
                f"data, make sure the data you provide is valid: {e}"
            )

    def _fit_transform_categorical_multi_features(self, X_multi):
        """Fit and transform multi-categorical features.

        Parameters
        ----------
        X_multi : pandas DataFrame
            The multi-categorical feature matrix.

        Returns
        -------
        pandas DataFrame
            The transformed multi-categorical feature matrix.

        Raises
        ------
        RuntimeError
            If failed to transform multi-categorical features.
        """
        check_sklearn_transformer_is_not_fitted(
            self.feature_cat_multi_encoder,
            "Failed to transform multi-categorical feature, the multi-categorical encoder is already fitted. "
            + "Create a new instance of DataTransformer to transform a new dataset.",
        )

        try:
            try:
                # try to set sparse_output=False for OneHotEncoder
                self.feature_cat_multi_encoder.set_params(sparse_output=False)
            except Exception as e:
                pass
            X_multi_ = self.feature_cat_multi_encoder.fit_transform(X_multi)
        except Exception as e:
            raise RuntimeError(
                f"Failed to transform multi-categorical feature with the provided encoder "
                f"({self.feature_cat_multi_encoder}), make sure the encoder you provide is valid: {e}"
            )
        # replace the sep char '_' with '=' in the one-hot encoded feature names
        onehot_columns = [
            col.replace("_", "=")
            for col in self.feature_cat_multi_encoder.get_feature_names_out(
                X_multi.columns
            )
        ]
        return pd.DataFrame(X_multi_, columns=onehot_columns, index=X_multi.index)

    def _inverse_transform_categorical_multi_features(self, X_multi):
        """Inverse transform multi-categorical features.

        Parameters
        ----------
        X_multi : pandas DataFrame
            The transformed multi-categorical feature matrix.

        Returns
        -------
        pandas DataFrame
            The inverse transformed multi-categorical feature matrix.

        Raises
        ------
        RuntimeError
            If failed to inverse transform multi-categorical features.
        """
        check_sklearn_transformer_is_fitted(
            self.feature_cat_multi_encoder,
            "Failed to inverse transform multi-categorical feature, the multi-categorical encoder is not fitted yet",
        )
        try:
            return self.feature_cat_multi_encoder.inverse_transform(X_multi)
        except Exception as e:
            raise ValueError(
                f"Failed to inverse transform multi-categorical feature with the provided data, make sure the data you provide is valid: {e}"
            )

    def _fit_transform_target_attr(self, y_raw, pos_value=None, verbose=False):
        """Fit and transform the target attribute.

        Parameters
        ----------
        y_raw : pandas Series
            The raw target attribute.

        pos_value : object or None, default=None
            The positive value of the target attribute.

        verbose : bool, default=False
            Whether to print verbose output.

        Returns
        -------
        pandas Series
            The transformed target attribute, with positive value encoded as 1
            and negative value encoded as 0.

        Raises
        ------
        RuntimeError
            If failed to transform the target attribute.
        """
        check_sklearn_transformer_is_not_fitted(
            self.target_encoder,
            "Failed to transform target attribute, the target_encoder is already fitted. "
            + "Create a new instance of DataTransformer to transform a new dataset.",
        )

        y_raw = check_target_attr(
            y_raw, accept_non_numerical=True
        )  # check if target is binary
        labels, counts = np.unique(y_raw, return_counts=True)

        # determine positive and negative values
        if pos_value is not None:
            # if positive value is specified
            assert (
                pos_value in labels
            ), f"`pos_value` should be one of the unique values ({labels}) in the target attribute, got {pos_value} instead"
        else:
            # if positive value is not specified, use the minority class as the positive class
            pos_value = labels[np.argmin(counts)]
            if verbose:
                print(
                    f"Positive value is not specified, using the minority class '{pos_value}' as the positive label"
                )
        neg_value = [x for x in labels if x != pos_value][0]

        # encode the target attribute and store the encoder
        categories = [neg_value, pos_value]
        self.target_encoder.set_params(
            categories=[categories]
        )  # set the categories for the encoder
        y = self.target_encoder.fit_transform(y_raw.reshape(-1, 1)).ravel()
        return y

    def _inverse_transform_target_attr(self, y):
        """Inverse transform the target attribute.

        Parameters
        ----------
        y : pandas Series
            The transformed target attribute. Must be numerical.

        Returns
        -------
        pandas Series
            The inverse transformed target attribute.
        """
        check_sklearn_transformer_is_fitted(
            self.target_encoder,
            "Failed to inverse transform target attribute, the target_encoder is not fitted yet",
        )
        y = check_target_attr(y, accept_non_numerical=False)
        y = self.target_encoder.inverse_transform(y.reshape(-1, 1)).ravel()
        return pd.Series(y).astype("category")

    def _fit_transform_sensitive_attr(
        self, s_raw, y=None, pos_value=None, verbose=False
    ):
        """Fit and transform the sensitive attribute.

        Parameters
        ----------
        s_raw : pandas Series
            The raw sensitive attribute.

        y : pandas Series or None, default=None
            The encoded target attribute. Used to determine the positive value
            of the sensitive attribute if `pos_value` is not specified. Ignored
            if `pos_value` is specified.

        pos_value : object or None, default=None
            The positive value of the sensitive attribute. If not specified,
            the advantageous group (with high positive label ratio) will be
            used as the positive value.

        verbose : bool, default=False
            Whether to print verbose output. Default is False.

        Returns
        -------
        pandas Series
            The transformed sensitive attribute. The positive value is encoded as 1
            and the negative value is encoded as 0.

        Raises
        ------
        RuntimeError
            If failed to transform the sensitive attribute.
        """
        check_sklearn_transformer_is_not_fitted(
            self.sensitive_encoder,
            "Failed to transform sensitive attribute, the sensitive_encoder is already fitted. "
            + "Create a new instance of DataTransformer to transform a new dataset.",
        )
        s_raw = check_sensitive_attr(
            s_raw, accept_non_numerical=True
        )  # check if sensitive attribute is binary
        memberships, counts = np.unique(s_raw, return_counts=True)

        # determine positive and negative values
        if pos_value is not None:
            # if positive value is specified
            assert (
                pos_value in memberships
            ), f"`pos_value` should be one of the unique values ({memberships}) in the sensitive attribute, got {pos_value} instead"
        else:
            # if positive value is not specified, use the advatageous group (with high positive label ratio) as the positive
            assert (
                y is not None
            ), "The encoded target `y` must be provided if `pos_value` is not specified inorder to encode the sensitive_attr based on positive ratio"
            y = check_target_attr(y, accept_non_numerical=False)
            pos_ratios = {
                m: np.sum(y[s_raw == m]) / np.sum(s_raw == m) for m in memberships
            }
            pos_value = max(pos_ratios, key=pos_ratios.get)
        neg_value = [x for x in memberships if x != pos_value][0]

        # encode the sensitive attribute, and store the encoder
        categories = [neg_value, pos_value]
        self.sensitive_encoder.set_params(
            categories=[categories]
        )  # set the categories for the encoder
        s = self.sensitive_encoder.fit_transform(s_raw.reshape(-1, 1)).ravel()
        return s

    def _inverse_transform_sensitive_attr(self, s):
        """Inverse transform the sensitive attribute.

        Parameters
        ----------
        s : pandas Series
            The transformed sensitive attribute. Must be numerical.

        Returns
        -------
        pandas Series
            The inverse transformed sensitive attribute.

        Raises
        ------
        ValueError
            If failed to inverse transform the sensitive attribute.
        """
        check_sklearn_transformer_is_fitted(
            self.sensitive_encoder,
            "Failed to inverse transform sensitive attribute, the sensitive_encoder is not fitted yet",
        )
        s = check_sensitive_attr(s, accept_non_numerical=False)
        s = self.sensitive_encoder.inverse_transform(s.reshape(-1, 1)).ravel()
        return pd.Series(s).astype("category")
