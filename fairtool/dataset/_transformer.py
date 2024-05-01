"""
This module implements the FairDataTransformer class.
"""

# %%
LOCAL_DEBUG = True

if not LOCAL_DEBUG:
    from ._check import *
    from ._utils import *
else:  # pragma: no cover
    # For local debugging purposes
    import sys

    sys.path.append("..")
    from dataset._check import *
    from dataset._utils import *

import sklearn
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
    feature_names_in_ : list of str
        The input feature names.

    feature_names_in_type_dict_ : dict
        The dictionary of input feature types. The keys are 'numerical',
        'categorical-binary', and 'categorical-multi', and the values are
        the corresponding feature names.

    feature_names_out_ : list of str
        The transformed feature names.

    feature_names_out_type_dict_ : dict
        The dictionary of transformed feature types. The keys are 'numerical',
        'categorical-binary', and 'categorical-multi', and the values are
        the corresponding feature names.

    target_name_ : str
        The target attribute name.

    sensitive_name_ : str
        The sensitive attribute name.

    X_out_ : pandas DataFrame
        The transformed feature matrix.

    y_out_ : pandas Series
        The transformed target attribute.

    s_out_ : pandas Series
        The transformed sensitive attribute.
    """

    def __init__(
        self,
        feature_num_scaler=None,
        feature_cat_bin_encoder=None,
        feature_cat_multi_encoder=None,
    ):
        """Initialize FairDataTransformer."""
        if feature_num_scaler is None:
            self._feature_num_scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
        else:
            scaler_classes = (
                sklearn.base.BaseEstimator,
                sklearn.base.TransformerMixin,
                sklearn.base.OneToOneFeatureMixin,
            )
            assert isinstance(feature_num_scaler, scaler_classes), (
                f"Invalid `feature_num_scaler`={feature_num_scaler}, "
                f"it should be a valid sklearn transformer instance "
                f"inheriting from {scaler_classes}."
            )
            self._feature_num_scaler = feature_num_scaler

        if feature_cat_bin_encoder is None:
            self._feature_cat_bin_encoder = OrdinalEncoder()
        else:
            encoder_class = sklearn.preprocessing._encoders._BaseEncoder
            assert isinstance(feature_cat_bin_encoder, encoder_class), (
                f"Invalid `feature_cat_bin_encoder`={feature_cat_bin_encoder}, "
                f"it should be a valid sklearn encoder instance inheriting from {encoder_class}."
            )
            self._feature_cat_bin_encoder = feature_cat_bin_encoder

        if feature_cat_multi_encoder is None:
            self._feature_cat_multi_encoder = OneHotEncoder()
        else:
            encoder_class = sklearn.preprocessing._encoders._BaseEncoder
            assert isinstance(feature_cat_multi_encoder, encoder_class), (
                f"Invalid `feature_cat_multi_encoder`={feature_cat_multi_encoder}, "
                f"it should be a valid sklearn encoder instance inheriting from {encoder_class}."
            )
            self._feature_cat_multi_encoder = feature_cat_multi_encoder

        self._target_encoder = OrdinalEncoder()
        self._sensitive_encoder = OrdinalEncoder()
        self._fitted = False

        return

    def fit(
        self,
        X,
        y,
        s,
        categorical_features=None,
        target_pos_value=None,
        sensitive_pos_value=None,
        verbose=False,
    ):
        """Fit the transformer.

        Parameters
        ----------
        X : pandas DataFrame
            The input feature matrix.

        y : pandas Series
            The input target attribute. Must be binary.

        s : pandas Series
            The input sensitive attribute. Must be binary.

        categorical_features : list of str or None, default=None
            The list of categorical feature names. If None, categorical
            features will be inferred from the DataFrame.

        target_pos_value : object or None, default=None
            The positive value of the target attribute. If not specified,
            the minority class (with smaller portion) will be used as the
            positive value.

        sensitive_pos_value : object or None, default=None
            The positive value of the sensitive attribute. If not specified,
            the advantageous group (with higher positive label ratio) will be
            used as the positive value.

        verbose : bool, default=False
            Whether to print verbose output.

        Raises
        ------
        ValueError
            If failed to fit the transformer.
        """
        X, y, s = check_X_y_s(X, y, s, accept_non_numerical=True)
        feature_names = X.columns.tolist()
        target_name = y.name
        sensitive_name = s.name

        X_without_s = X.drop(columns=[sensitive_name])
        data = pd.concat([X_without_s, y, s], axis=1)
        check_feature_target_sensitive_names(
            data, feature_names, target_name, sensitive_name, accept_non_numerical=True
        )
        # set the feature dtypes (automatically determined if categorical_features=None)
        X = set_feature_dtypes(
            X, categorical_features=categorical_features, verbose=verbose
        )

        # parse the feature dtypes, keys are 'numerical', 'categorical-binary', 'categorical-multi'
        dtype_features = parse_detailed_feature_dtypes(
            X, feature_names=feature_names, dtype_as_key=True, verbose=verbose
        )
        # rearrange raw features in the order of numerical, binary categorical, and multi-categorical
        feature_names = (
            dtype_features["numerical"]
            + dtype_features["categorical-binary"]
            + dtype_features["categorical-multi"]
        )
        self.feature_names_ = feature_names
        self.target_name_ = target_name
        self.sensitive_name_ = sensitive_name
        # store the data with rearranged columns
        self.data_raw_ = data[
            [f for f in feature_names if f != sensitive_name]
            + [sensitive_name, target_name]
        ]

        # fit transformers
        self._fit_y(y, pos_value=target_pos_value, verbose=verbose)
        self._fit_s(
            s,
            y_encoded=self.transform_y(y, return_pandas=False),
            pos_value=sensitive_pos_value,
            verbose=verbose,
        )
        self._fit_X(X, dtype_features, sensitive_name)
        self._fitted = True

        # store the input feature names and their types
        self.feature_names_in_ = feature_names
        self.feature_names_in_type_dict_ = dtype_features
        assert set(self.feature_names_in_) == set(
            np.concatenate(list(self.feature_names_in_type_dict_.values()))
        ), f"Feature set mismatch between self.feature_names_in_ and self.feature_names_in_type_dict_."

        # store the output feature names and their types
        self.feature_names_out_type_dict_ = {
            "numerical": self._feature_num_scaler.get_feature_names_out(
                input_features=dtype_features["numerical"]
            ),
            "categorical-binary": self._feature_cat_bin_encoder.get_feature_names_out(
                input_features=dtype_features["categorical-binary"]
            ),
            "categorical-multi": self._feature_cat_multi_encoder.get_feature_names_out(
                input_features=dtype_features["categorical-multi"]
            ),
        }
        self.feature_names_out_ = np.concatenate(
            list(self.feature_names_out_type_dict_.values())
        )
        assert set(self.feature_names_out_) == set(
            np.concatenate(list(self.feature_names_out_type_dict_.values()))
        ), f"Feature set mismatch between self.feature_names_out_type_dict_ and self.feature_names_out_."

    def transform(self, X, y, s, return_pandas=True):
        """Transform the dataset.

        Parameters
        ----------
        X : pandas DataFrame
            The input feature matrix.

        y : pandas Series
            The input target attribute.

        s : pandas Series
            The input sensitive attribute.

        return_pandas : bool, default=True
            Whether to return the values as pandas DataFrame or Series, default is True.
            If False, the values will be returned as numpy arrays.

        Returns
        -------
        X : pandas DataFrame or numpy array
            The transformed feature matrix.

        y : pandas Series or numpy array
            The transformed target attribute.

        s : pandas Series or numpy array
            The transformed sensitive attribute.
        """
        check_sklearn_transformer_is_fitted(
            self,
            (
                f"Failed to transform the dataset, the transformer is not fitted yet. "
                f"Fit the transformer before calling `transform`."
            ),
        )
        X_ = self._transform_X(X, self.feature_names_in_type_dict_, return_pandas=True)
        y_ = self.transform_y(y, return_pandas=True)
        s_ = self.transform_s(s, return_pandas=True)

        # store the output feature names and their types
        assert np.all(
            X_.columns == self.feature_names_out_
        ), f"Feature set mismatch between self.feature_names_out_ and the transformed feature matrix."

        self.X_out_ = X_
        self.y_out_ = y_
        self.s_out_ = s_

        if return_pandas:
            return X_, y_, s_
        else:
            return X_.values, y_.values, s_.values

    def fit_transform(
        self,
        X,
        y,
        s,
        categorical_features=None,
        target_pos_value=None,
        sensitive_pos_value=None,
        return_pandas=True,
        verbose=False,
    ):
        """Fit and transform the dataset.

        Parameters
        ----------
        X : pandas DataFrame
            The input feature matrix.

        y : pandas Series
            The input target attribute. Must be binary.

        s : pandas Series
            The input sensitive attribute. Must be binary.

        categorical_features : list of str or None, default=None
            The list of categorical feature names. If None, categorical
            features will be inferred from the DataFrame.

        target_pos_value : object or None, default=None
            The positive value of the target attribute. If not specified,
            the minority class (with smaller portion) will be used as the
            positive value.

        sensitive_pos_value : object or None, default=None
            The positive value of the sensitive attribute. If not specified,
            the advantageous group (with higher positive label ratio) will be
            used as the positive value.

        return_pandas : bool, default=True
            Whether to return the values as pandas DataFrame or Series, default is True.
            If False, the values will be returned as numpy arrays.

        verbose : bool, default=False
            Whether to print verbose output.

        Returns
        -------
        X : pandas DataFrame or numpy array
            The transformed feature matrix.

        y : pandas Series or numpy array
            The transformed target attribute.

        s : pandas Series or numpy array
            The transformed sensitive attribute.
        """
        self.fit(
            X,
            y,
            s,
            categorical_features,
            target_pos_value,
            sensitive_pos_value,
            verbose,
        )
        return self.transform(X, y, s, return_pandas=return_pandas)

    def fit_transform_from_single_dataframe(
        self,
        data,
        target_name,
        sensitive_name,
        feature_names=None,
        categorical_features=None,
        exclude_sensitive_feature=False,
        target_pos_value=None,
        sensitive_pos_value=None,
        return_pandas=True,
        verbose=False,
    ):
        if feature_names is None:
            feature_names = data.columns.tolist()
            feature_names.remove(target_name)
            if exclude_sensitive_feature:
                feature_names.remove(sensitive_name)
        check_feature_target_sensitive_names(
            data, feature_names, target_name, sensitive_name, accept_non_numerical=True
        )
        X, y, s = data[feature_names], data[target_name], data[sensitive_name]

        self.fit(
            X=X,
            y=y,
            s=s,
            categorical_features=categorical_features,
            target_pos_value=target_pos_value,
            sensitive_pos_value=sensitive_pos_value,
            verbose=verbose,
        )
        return self.transform(X, y, s, return_pandas)

    def get_feature_names_out(self):
        """Get the transformed feature names.

        Returns
        -------
        list of str
            The transformed feature names.
        """
        assert (
            self._fitted
        ), f"The transformer is not fitted. Fit the transformer before accessing `get_feature_names_out`."
        return self.feature_names_out_

    def _fit_X(self, X, dtype_features, sensitive_name):
        """Fit the transformer for the feature matrix.

        Parameters
        ----------
        X : pandas DataFrame
            The feature matrix.

        dtype_features : dict
            The dictionary of feature dtypes. The keys should be 'numerical',
            'categorical-binary', and 'categorical-multi', and the values should
            be the corresponding feature names.

        sensitive_name : str
            The name of the sensitive attribute column. Used to guarantee
            consistent encoding for the sensitive attribute in X and s.
        """
        self._fit_numerical_features(X[dtype_features["numerical"]])
        self._fit_categorical_binary_features(
            X[dtype_features["categorical-binary"]], sensitive_name
        )
        self._fit_categorical_multi_features(X[dtype_features["categorical-multi"]])
        return

    def _transform_X(self, X, dtype_features, return_pandas=True):
        """Transform the feature matrix.

        Parameters
        ----------
        X : pandas DataFrame
            The feature matrix.

        dtype_features : dict
            The dictionary of feature dtypes. The keys should be 'numerical',
            'categorical-binary', and 'categorical-multi', and the values should
            be the corresponding feature names.

        return_pandas : bool, default=True
            Whether to return the transformed features as a pandas DataFrame.
            Default is True. If False, the transformed features will be returned
            as numpy arrays.

        Returns
        -------
        X_ : pandas DataFrame or numpy array
            The transformed feature matrix.

        Raises
        ------
        RuntimeError
            If failed to transform the feature matrix.
        """
        X_num_ = self._transform_numerical_features(
            X[dtype_features["numerical"]], return_pandas=return_pandas
        )
        X_bin_ = self._transform_categorical_binary_features(
            X[dtype_features["categorical-binary"]], return_pandas=return_pandas
        )
        X_multi_ = self._transform_categorical_multi_features(
            X[dtype_features["categorical-multi"]], return_pandas=return_pandas
        )
        if return_pandas:
            X_ = pd.concat([X_num_, X_bin_, X_multi_], axis=1)
        else:
            X_ = np.concatenate([X_num_, X_bin_, X_multi_], axis=1)
        return X_

    def _fit_transform_X(self, X, dtype_features, sensitive_name, return_pandas=True):
        """Fit and transform the feature matrix.

        Parameters
        ----------
        X : pandas DataFrame
            The feature matrix.

        dtype_features : dict
            The dictionary of feature dtypes. The keys should be 'numerical',
            'categorical-binary', and 'categorical-multi', and the values should
            be the corresponding feature names.

        sensitive_name : str
            The name of the sensitive attribute column. Used to guarantee
            consistent encoding for the sensitive attribute in X and s.

        return_pandas : bool, default=True
            Whether to return the transformed features as a pandas DataFrame.
            Default is True. If False, the transformed features will be returned
            as numpy arrays.

        Returns
        -------
        X_ : pandas DataFrame or numpy array
            The transformed feature matrix.

        Raises
        ------
        RuntimeError
            If failed to transform the feature matrix.
        """
        self._fit_X(X, dtype_features, sensitive_name)
        return self._transform_X(X, dtype_features, return_pandas=return_pandas)

    def inverse_transform_X(self, X, return_pandas=True):
        """Inverse transform the features.

        Parameters
        ----------
        X : pandas DataFrame
            The transformed feature matrix.

        return_pandas : bool, default=True
            Whether to return the transformed features as a pandas DataFrame.
            Default is True. If False, the transformed features will be returned
            as numpy arrays.

        Returns
        -------
        X_ : pandas DataFrame or numpy array
            The inverse transformed feature matrix.
        """
        assert self._fitted, (
            f"The transformer is not fitted, call `fit_transform_X_y_s_from_dataframe` "
            f"before calling `inverse_transform_X`."
        )
        assert isinstance(
            X, pd.DataFrame
        ), f"Invalid input `X`={X}, it should be a pandas DataFrame."
        assert set(X.columns) == set(self.feature_names_out_), (
            f"To inverse transform the features, the input DataFrame should "
            f"have columns names consistent with the transformed feature names.\n"
            f"Expected feature names: {self.feature_names_out_}."
        )

        in_feats = self.feature_names_in_type_dict_
        out_feats = self.feature_names_out_type_dict_

        X_num_ = self._inverse_transform_numerical_features(
            X[out_feats["numerical"]], return_pandas=return_pandas
        )
        X_bin_ = self._inverse_transform_categorical_binary_features(
            X[out_feats["categorical-binary"]], return_pandas=return_pandas
        )
        X_multi_ = self._inverse_transform_categorical_multi_features(
            X[out_feats["categorical-multi"]], return_pandas=return_pandas
        )
        if return_pandas:
            return pd.concat([X_num_, X_bin_, X_multi_], axis=1)
        else:
            return np.concatenate([X_num_, X_bin_, X_multi_], axis=1)

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
        ), f"The transformer is not fitted. Fit the transformer before accessing `X_in_`."
        return self.data_raw_[self.feature_names_in_]

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
        ), f"The transformer is not fitted. Fit the transformer before accessing `y_in_`."
        return self.data_raw_[self.target_name_]

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
        ), f"The transformer is not fitted. Fit the transformer before accessing `s_in_`."
        return self.data_raw_[self.sensitive_name_]

    def _fit_numerical_features(self, X_num):
        """Fit the numerical features.

        Parameters
        ----------
        X_num : pandas DataFrame
            The numerical feature matrix.

        Raises
        ------
        RuntimeError
            If failed to transform numerical features.
        """
        check_sklearn_transformer_is_not_fitted(
            self._feature_num_scaler,
            (
                f"A numerical feature scaler is already fitted, create a new instance "
                f"of DataTransformer to transform a new dataset"
            ),
        )
        try:
            self._feature_num_scaler.fit(X_num)
        except Exception as e:
            raise RuntimeError(
                f"Failed to fit numerical feature with the provided "
                f"feature_num_scaler={self._feature_num_scaler}, make sure "
                f"the scaler you provide is valid.\nError: {e}."
            )

    def _transform_numerical_features(self, X_num, return_pandas=True):
        """Transform the numerical features.

        Parameters
        ----------
        X_num : pandas DataFrame
            The numerical feature matrix.

        return_pandas : bool, default=True
            Whether to return the transformed features as a pandas DataFrame.
            Default is True. If False, the transformed features will be returned
            as numpy arrays.

        Returns
        -------
        X_num_ : pandas DataFrame or numpy array
            The transformed numerical feature matrix.

        Raises
        ------
        RuntimeError
            If failed to transform numerical features.
        """
        check_sklearn_transformer_is_fitted(
            self._feature_num_scaler,
            (
                f"Failed to transform numerical feature, the numerical feature scaler is not fitted yet."
            ),
        )
        assert isinstance(
            X_num, pd.DataFrame
        ), f"Invalid input `X_num`={X_num}, it should be a pandas DataFrame."
        assert isinstance(
            return_pandas, bool
        ), f"Invalid input `return_pandas`={return_pandas}, it should be a boolean."

        try:
            X_num_ = self._feature_num_scaler.transform(X_num)
            if return_pandas:
                columns = self._feature_num_scaler.get_feature_names_out()
                return pd.DataFrame(X_num_, columns=columns, index=X_num.index)
            else:
                return X_num_
        except Exception as e:
            raise RuntimeError(
                f"Failed to transform numerical feature with the provided "
                f"feature_num_scaler={self._feature_num_scaler}, make sure "
                f"the data you provide is valid.\nError: {e}."
            )

    def _fit_transform_numerical_features(self, X_num, return_pandas=True):
        """Fit and transform the numerical features.

        Parameters
        ----------
        X_num : pandas DataFrame
            The numerical feature matrix.

        return_pandas : bool, default=True
            Whether to return the transformed features as a pandas DataFrame.
            Default is True. If False, the transformed features will be returned
            as numpy arrays.

        Returns
        -------
        X_num_ : pandas DataFrame or numpy array
            The transformed numerical feature matrix.

        Raises
        ------
        RuntimeError
            If failed to transform numerical features.
        """
        self._fit_numerical_features(X_num)
        return self._transform_numerical_features(X_num, return_pandas=return_pandas)

    def _inverse_transform_numerical_features(self, X_num, return_pandas=True):
        """Inverse transform numerical features.

        Parameters
        ----------
        X_num : array-like
            The transformed numerical feature matrix.

        return_pandas : bool, default=True
            Whether to return the transformed features as a pandas DataFrame.
            Default is True. If False, the transformed features will be returned
            as numpy arrays.

        Returns
        -------
        X_num_ : pandas DataFrame or numpy array
            The inverse transformed numerical feature matrix.

        Raises
        ------
        RuntimeError
            If failed to inverse transform numerical features.
        """
        check_sklearn_transformer_is_fitted(
            self._feature_num_scaler,
            (
                f"Failed to inverse transform numerical feature, "
                f"the numerical feature scaler is not fitted yet."
            ),
        )
        assert isinstance(
            return_pandas, bool
        ), f"Invalid input `return_pandas`={return_pandas}, it should be a boolean."
        try:
            X_num_ = self._feature_num_scaler.inverse_transform(X_num)
            if return_pandas:
                columns = self._feature_num_scaler.feature_names_in_
                return pd.DataFrame(X_num_, columns=columns, index=X_num.index)
            else:
                return X_num_
        except Exception as e:
            raise RuntimeError(
                f"Failed to inverse transform numerical feature with the "
                f"provided data, make sure the data you provide is valid.\nError: {e}."
            )

    def _fit_categorical_binary_features(self, X_bin, sensitive_name):
        """Fit the encoder for binary categorical features.

        Parameters
        ----------
        X_bin : pandas DataFrame
            The binary categorical feature matrix.

        sensitive_name : str
            The name of the sensitive attribute column. The encoding rule for
            the sensitive attribute will be used in a consistent manner if the
            sensitive attribute is included in the features.

        Raises
        ------
        RuntimeError
            If failed to transform binary categorical features.
        """
        check_sklearn_transformer_is_not_fitted(
            self._feature_cat_bin_encoder,
            (
                f"Failed to fit the binary categorical feature encoder, the binary categorical encoder "
                f"is already fitted. Create a new instance of DataTransformer to transform a new dataset."
            ),
        )
        categories = []
        for col in X_bin.columns:
            if col == sensitive_name:
                check_sklearn_transformer_is_fitted(
                    self._sensitive_encoder,
                    (
                        f"The sensitive attribute is included in the features, but the sensitive_encoder "
                        f"is not fitted yet. Check the order of the transformation steps to guarantee "
                        f"consistent encoding for the sensitive feature."
                    ),
                )
                categories.append(self._sensitive_encoder.categories_[0])
            else:
                categories.append(X_bin[col].unique())

        try:
            self._feature_cat_bin_encoder.set_params(categories=categories)
            self._feature_cat_bin_encoder.fit(X_bin)
        except Exception as e:
            raise RuntimeError(
                f"Failed to fit binary categorical feature encoder ({self._feature_cat_bin_encoder}), "
                f"make sure the encoder you provide is valid.\nError: {e}."
            )

    def _transform_categorical_binary_features(self, X_bin, return_pandas=True):
        """Transform binary categorical features.

        Parameters
        ----------
        X_bin : pandas DataFrame
            The binary categorical feature matrix.

        return_pandas : bool, default=True
            Whether to return the transformed features as a pandas DataFrame.
            Default is True. If False, the transformed features will be returned
            as numpy arrays.

        Returns
        -------
        X_bin_ : pandas DataFrame or numpy array
            The transformed binary categorical feature matrix.

        Raises
        ------
        RuntimeError
            If failed to transform binary categorical features.
        """
        check_sklearn_transformer_is_fitted(
            self._feature_cat_bin_encoder,
            (
                f"Failed to transform binary categorical feature, the binary categorical encoder is not fitted yet."
            ),
        )
        assert isinstance(
            return_pandas, bool
        ), f"Invalid input `return_pandas`={return_pandas}, it should be a boolean."
        try:
            X_bin_ = self._feature_cat_bin_encoder.transform(X_bin)
            if return_pandas:
                columns = self._feature_cat_bin_encoder.get_feature_names_out()
                return pd.DataFrame(X_bin_, columns=columns, index=X_bin.index)
            else:
                return X_bin_
        except Exception as e:
            raise RuntimeError(
                f"Failed to transform binary categorical feature with the provided encoder "
                f"({self._feature_cat_bin_encoder}), make sure the encoder you provide is valid.\nError: {e}."
            )

    def _fit_transform_categorical_binary_features(
        self, X_bin, sensitive_name, return_pandas=True
    ):
        """Fit and transform binary categorical features.

        Parameters
        ----------
        X_bin : pandas DataFrame
            The binary categorical feature matrix.

        sensitive_name : str
            The name of the sensitive attribute column. The encoding rule for
            the sensitive attribute will be used in a consistent manner if the
            sensitive attribute is included in the features.

        return_pandas : bool, default=True
            Whether to return the transformed features as a pandas DataFrame.
            Default is True. If False, the transformed features will be returned
            as numpy arrays.

        Returns
        -------
        X_bin_ : pandas DataFrame or numpy array
            The transformed binary categorical feature matrix.

        Raises
        ------
        RuntimeError
            If failed to transform binary categorical features.

        """
        self._fit_categorical_binary_features(X_bin, sensitive_name)
        return self._transform_categorical_binary_features(
            X_bin, return_pandas=return_pandas
        )

    def _inverse_transform_categorical_binary_features(self, X_bin, return_pandas=True):
        """Inverse transform binary categorical features.

        Parameters
        ----------
        X_bin : pandas DataFrame
            The transformed binary categorical feature matrix.

        return_pandas : bool, default=True
            Whether to return the transformed features as a pandas DataFrame.
            Default is True. If False, the transformed features will be returned
            as numpy arrays.

        Returns
        -------
        X_bin_ : pandas DataFrame or numpy array
            The inverse transformed binary categorical feature matrix.

        Raises
        ------
        RuntimeError
            If failed to inverse transform binary categorical features.
        """
        check_sklearn_transformer_is_fitted(
            self._feature_cat_bin_encoder,
            (
                f"Failed to inverse transform binary categorical feature, "
                f"the binary categorical encoder is not fitted yet"
            ),
        )
        try:
            X_bin_ = self._feature_cat_bin_encoder.inverse_transform(X_bin)
            if return_pandas:
                columns = self._feature_cat_bin_encoder.feature_names_in_
                return pd.DataFrame(X_bin_, columns=columns, index=X_bin.index)
            else:
                return X_bin_
        except Exception as e:
            raise RuntimeError(
                f"Failed to inverse transform binary categorical feature with the provided "
                f"encoder ({self._feature_cat_bin_encoder}), got error.\nError: {e}."
            )

    def _fit_categorical_multi_features(self, X_multi):
        """Fit the encoder for multi-categorical features.

        Parameters
        ----------
        X_multi : pandas DataFrame
            The multi-categorical feature matrix.

        Raises
        ------
        RuntimeError
            If failed to transform multi-categorical features.
        """
        check_sklearn_transformer_is_not_fitted(
            self._feature_cat_multi_encoder,
            (
                f"Failed to fit the multi-categorical feature encoder, the multi-categorical encoder "
                f"is already fitted. Create a new instance of DataTransformer to transform a new dataset."
            ),
        )
        try:
            self._feature_cat_multi_encoder.set_params(sparse_output=False)
            self._feature_cat_multi_encoder.fit(X_multi)
        except Exception as e:
            raise RuntimeError(
                f"Failed to fit multi-categorical feature with the provided encoder "
                f"{self._feature_cat_multi_encoder}, got error.\nError: {e}."
            )

    def _transform_categorical_multi_features(self, X_multi, return_pandas=True):
        """Transform multi-categorical features with the fitted feature_cat_multi_encoder.

        Parameters
        ----------
        X_multi : pandas DataFrame
            The multi-categorical feature matrix.

        return_pandas : bool, default=True
            Whether to return the transformed features as a pandas DataFrame.
            Default is True. If False, the transformed features will be returned
            as numpy arrays.

        Returns
        -------
        X_multi_ : pandas DataFrame or numpy array
            The transformed multi-categorical feature matrix.

        Raises
        ------
        RuntimeError
            If failed to transform multi-categorical features.
        """
        check_sklearn_transformer_is_fitted(
            self._feature_cat_multi_encoder,
            (
                f"Failed to transform multi-categorical feature, the multi-categorical encoder is not fitted yet."
            ),
        )
        try:
            X_multi_ = self._feature_cat_multi_encoder.transform(X_multi)
            if return_pandas:
                columns = self._feature_cat_multi_encoder.get_feature_names_out()
                return pd.DataFrame(X_multi_, columns=columns, index=X_multi.index)
            else:
                return X_multi_
        except Exception as e:
            raise RuntimeError(
                f"Failed to transform multi-categorical feature with the provided encoder "
                f"{self._feature_cat_multi_encoder}, got error:\n{e}."
            )

    def _fit_transform_categorical_multi_features(self, X_multi, return_pandas=True):
        """Fit and transform multi-categorical features.

        Parameters
        ----------
        X_multi : pandas DataFrame
            The multi-categorical feature matrix.

        return_pandas : bool, default=True
            Whether to return the transformed features as a pandas DataFrame.
            Default is True. If False, the transformed features will be returned
            as numpy arrays.

        Returns
        -------
        X_multi_ : pandas DataFrame or numpy array
            The transformed multi-categorical feature matrix.

        Raises
        ------
        RuntimeError
            If failed to transform multi-categorical features.
        """
        self._fit_categorical_multi_features(X_multi)
        return self._transform_categorical_multi_features(
            X_multi, return_pandas=return_pandas
        )

    def _inverse_transform_categorical_multi_features(
        self, X_multi, return_pandas=True
    ):
        """Inverse transform multi-categorical features.

        Parameters
        ----------
        X_multi : pandas DataFrame
            The transformed multi-categorical feature matrix.

        return_pandas : bool, default=True
            Whether to return the transformed features as a pandas DataFrame.
            Default is True. If False, the transformed features will be returned
            as numpy arrays.

        Returns
        -------
        X_multi_ : pandas DataFrame or numpy array
            The inverse transformed multi-categorical feature matrix.

        Raises
        ------
        RuntimeError
            If failed to inverse transform multi-categorical features.
        """
        check_sklearn_transformer_is_fitted(
            self._feature_cat_multi_encoder,
            (
                f"Failed to inverse transform multi-categorical feature, "
                f"the multi-categorical encoder is not fitted yet."
            ),
        )
        try:
            X_multi_ = self._feature_cat_multi_encoder.inverse_transform(X_multi)
            if return_pandas:
                columns = self._feature_cat_multi_encoder.feature_names_in_
                return pd.DataFrame(X_multi_, columns=columns, index=X_multi.index)
            else:
                return X_multi_
        except Exception as e:
            raise ValueError(
                f"Failed to inverse transform multi-categorical feature with "
                f"the provided data, make sure the data you provide is valid:\n{e}."
            )

    def _fit_y(self, y_raw, pos_value=None, verbose=False):
        """Fit the target attribute.

        Parameters
        ----------
        y_raw : pandas Series
            The raw target attribute.

        pos_value : object or None, default=None
            The positive value of the target attribute. If not specified, the
            minority class (with smaller portion) will be used as the positive value.

        verbose : bool, default=False
            Whether to print verbose output.

        Raises
        ------
        RuntimeError
            If failed to transform the target attribute.
        """
        check_sklearn_transformer_is_not_fitted(
            self._target_encoder,
            (
                f"Failed to transform target attribute, the target_encoder is already fitted. "
                f"Create a new instance of DataTransformer to transform a new dataset."
            ),
        )

        y_raw = check_target_attr(
            y_raw, accept_non_numerical=True
        )  # check if target is binary
        labels, counts = np.unique(y_raw, return_counts=True)

        # determine positive and negative values
        if pos_value is not None:
            # if positive value is specified
            assert pos_value in labels, (
                f"`pos_value` should be one of the unique values ({labels}) in the target attribute, "
                f"got {pos_value} instead."
            )
        else:
            # if positive value is not specified, use the minority class as the positive class
            pos_value = labels[np.argmin(counts)]
            if verbose:
                print(
                    f"Target positive value is not specified, using the minority class "
                    f"['{pos_value}'] as the positive class."
                )
        neg_value = [x for x in labels if x != pos_value][0]

        # encode the target attribute and store the encoder
        categories = [neg_value, pos_value]
        self._target_encoder.set_params(
            categories=[categories]
        )  # set the categories for the encoder

        # fit the encoder
        self._target_encoder.fit(y_raw.reshape(-1, 1))

    def transform_y(self, y, return_pandas=True):
        """Transform the target attribute.

        Parameters
        ----------
        y : pandas Series
            The raw target attribute.

        return_pandas : bool, default=True
            Whether to return the transformed target attribute as a pandas Series.
            Default is True. If False, the transformed target attribute will be
            returned as a numpy array.

        Returns
        -------
        y_ : pandas Series or numpy array
            The transformed target attribute, with positive value encoded as 1
            and negative value encoded as 0.

        Raises
        ------
        RuntimeError
            If failed to transform the target attribute.
        """
        check_sklearn_transformer_is_fitted(
            self._target_encoder,
            (
                f"Failed to transform target attribute, the target_encoder is not fitted yet."
            ),
        )
        y = check_target_attr(y, accept_non_numerical=True)
        try:
            y_ = self._target_encoder.transform(y.reshape(-1, 1)).ravel()
            if return_pandas:
                return pd.Series(y_, name=self.target_name_).astype("int")
            else:
                return y_
        except Exception as e:
            raise RuntimeError(
                f"Failed to transform target attribute with the provided encoder "
                f"({self._target_encoder}), make sure the encoder you provide is valid.\nError: {e}."
            )

    def _fit_transform_y(
        self, y_raw, pos_value=None, return_pandas=True, verbose=False
    ):
        """Fit and transform the target attribute.

        Parameters
        ----------
        y_raw : pandas Series
            The raw target attribute.

        pos_value : object or None, default=None
            The positive value of the target attribute. If not specified, the
            minority class (with smaller portion) will be used as the positive value.

        return_pandas : bool, default=True
            Whether to return the transformed target attribute as a pandas Series.
            Default is True. If False, the transformed target attribute will be
            returned as a numpy array.

        verbose : bool, default=False
            Whether to print verbose output.

        Returns
        -------
        y_ : pandas Series or numpy array
            The transformed target attribute, with positive value encoded as 1
            and negative value encoded as 0.

        Raises
        ------
        RuntimeError
            If failed to transform the target attribute.
        """
        self._fit_y(y_raw, pos_value=pos_value, verbose=verbose)
        return self.transform_y(y_raw, return_pandas=return_pandas)

    def inverse_transform_y(self, y, return_pandas=True):
        """Inverse transform the target attribute.

        Parameters
        ----------
        y : pandas Series
            The transformed target attribute. Must be numerical.

        return_pandas : bool, default=True
            Whether to return the transformed target attribute as a pandas Series.
            Default is True. If False, the transformed target attribute will be
            returned as a numpy array.

        Returns
        -------
        y_ : pandas Series or numpy array
            The inverse transformed target attribute.
        """
        check_sklearn_transformer_is_fitted(
            self._target_encoder,
            (
                f"Failed to inverse transform target attribute, the target_encoder is not fitted yet."
            ),
        )
        y = check_target_attr(y, accept_non_numerical=False)
        try:
            y_ = self._target_encoder.inverse_transform(y.reshape(-1, 1)).ravel()
            if return_pandas:
                return pd.Series(y_, name=self.target_name_).astype("category")
            else:
                return y_
        except Exception as e:
            raise RuntimeError(
                f"Failed to inverse transform target attribute with the provided "
                f"data, make sure the data you provide is valid.\nError: {e}."
            )

    def _fit_s(self, s_raw, y_encoded=None, pos_value=None, verbose=False):
        """Fit the sensitive attribute.

        Parameters
        ----------
        s_raw : pandas Series
            The raw sensitive attribute.

        y_encoded : pandas Series or None, default=None
            The encoded target attribute. Used to determine the positive value
            of the sensitive attribute if `pos_value` is not specified. Ignored
            if `pos_value` is specified.

        pos_value : object or None, default=None
            The positive value of the sensitive attribute. If not specified,
            the advantageous group (with high positive label ratio) will be
            used as the positive value.

        verbose : bool, default=False
            Whether to print verbose output.

        Raises
        ------
        RuntimeError
            If failed to transform the sensitive attribute.
        """
        check_sklearn_transformer_is_not_fitted(
            self._sensitive_encoder,
            (
                f"Failed to transform sensitive attribute, the sensitive_encoder is already fitted. "
                f"Create a new instance of DataTransformer to transform a new dataset."
            ),
        )
        s_raw = check_sensitive_attr(
            s_raw, accept_non_numerical=True
        )  # check if sensitive attribute is binary
        memberships, counts = np.unique(s_raw, return_counts=True)

        # determine positive and negative values
        if pos_value is not None:
            # if positive value is specified
            assert pos_value in memberships, (
                f"`pos_value` should be one of the unique values ({memberships}) in the sensitive attribute, "
                f"got {pos_value} instead."
            )
        else:
            # if positive value is not specified
            # use the advatageous group (with high positive label ratio) as the positive
            try:
                y_encoded = check_target_attr(y_encoded, accept_non_numerical=False)
            except Exception as e:
                raise ValueError(
                    f"Failed to determine the positive value of the sensitive attribute, "
                    f"the encoded target `y` must be provided if `pos_value` is not specified.\nError: {e}."
                )
            pos_ratios = {
                m: np.sum(y_encoded[s_raw == m]) / np.sum(s_raw == m)
                for m in memberships
            }
            pos_value = max(pos_ratios, key=pos_ratios.get)
            if verbose:
                print(
                    f"Sensitive positive value is not specified, using the advantageous group "
                    f"['{pos_value}'] with higher positive label ratio as the positive class."
                )
        neg_value = [x for x in memberships if x != pos_value][0]

        # encode the sensitive attribute, and store the encoder
        categories = [neg_value, pos_value]
        self._sensitive_encoder.set_params(
            categories=[categories]
        )  # set the categories for the encoder

        # fit the encoder
        try:
            self._sensitive_encoder.fit(s_raw.reshape(-1, 1))
        except Exception as e:
            raise RuntimeError(
                f"Failed to fit sensitive attribute with the provided encoder "
                f"({self._sensitive_encoder}), make sure the encoder you provide is valid.\nError: {e}."
            )

    def transform_s(self, s, return_pandas=True):
        """Transform the sensitive attribute.

        Parameters
        ----------
        s : pandas Series
            The raw sensitive attribute.

        return_pandas : bool, default=True
            Whether to return the transformed sensitive attribute as a pandas Series.
            Default is True. If False, the transformed sensitive attribute will be
            returned as a numpy array.

        Returns
        -------
        s_ : pandas Series or numpy array
            The transformed sensitive attribute. The positive value is encoded as 1
            and the negative value is encoded as 0.

        Raises
        ------
        RuntimeError
            If failed to transform the sensitive attribute.
        """
        check_sklearn_transformer_is_fitted(
            self._sensitive_encoder,
            (
                f"Failed to transform sensitive attribute, the sensitive_encoder is not fitted yet."
            ),
        )
        s = check_sensitive_attr(s, accept_non_numerical=True)
        try:
            s_ = self._sensitive_encoder.transform(s.reshape(-1, 1)).ravel()
            if return_pandas:
                return pd.Series(s_, name=self.sensitive_name_).astype("int")
            else:
                return s_
        except Exception as e:
            raise RuntimeError(
                f"Failed to transform sensitive attribute with the provided encoder "
                f"({self._sensitive_encoder}), make sure the encoder you provide is valid.\nError: {e}."
            )

    def _fit_transform_s(
        self, s_raw, y_encoded=None, pos_value=None, return_pandas=True, verbose=False
    ):
        """Fit and transform the sensitive attribute.

        Parameters
        ----------
        s_raw : pandas Series
            The raw sensitive attribute.

        y_encoded : pandas Series or None, default=None
            The encoded target attribute. Used to determine the positive value
            of the sensitive attribute if `pos_value` is not specified. Ignored
            if `pos_value` is specified.

        pos_value : object or None, default=None
            The positive value of the sensitive attribute. If not specified,
            the advantageous group (with high positive label ratio) will be
            used as the positive value.

        return_pandas : bool, default=True
            Whether to return the transformed sensitive attribute as a pandas Series.
            Default is True. If False, the transformed sensitive attribute will be
            returned as a numpy array.

        verbose : bool, default=False
            Whether to print verbose output. Default is False.

        Returns
        -------
        s_ : pandas Series or numpy array
            The transformed sensitive attribute. The positive value is encoded as 1
            and the negative value is encoded as 0.

        Raises
        ------
        RuntimeError
            If failed to transform the sensitive attribute.
        """
        self._fit_s(s_raw, y_encoded=y_encoded, pos_value=pos_value, verbose=verbose)
        return self.transform_s(s_raw, return_pandas=return_pandas)

    def inverse_transform_s(self, s, return_pandas=True):
        """Inverse transform the sensitive attribute.

        Parameters
        ----------
        s : pandas Series
            The transformed sensitive attribute. Must be numerical.

        return_pandas : bool, default=True
            Whether to return the transformed sensitive attribute as a pandas Series.
            Default is True. If False, the transformed sensitive attribute will be
            returned as a numpy array.

        Returns
        -------
        s_ : pandas Series or numpy array
            The inverse transformed sensitive attribute.

        Raises
        ------
        ValueError
            If failed to inverse transform the sensitive attribute.
        """
        check_sklearn_transformer_is_fitted(
            self._sensitive_encoder,
            "Failed to inverse transform sensitive attribute, the sensitive_encoder is not fitted yet.",
        )
        s = check_sensitive_attr(s, accept_non_numerical=False)
        try:
            s_ = self._sensitive_encoder.inverse_transform(s.reshape(-1, 1)).ravel()
            if return_pandas:
                return pd.Series(s_, name=self.sensitive_name_).astype("category")
            else:
                return s_
        except Exception as e:
            raise ValueError(
                f"Failed to inverse transform sensitive attribute with the provided "
                f"data, make sure the data you provide is valid.\nError: {e}."
            )
