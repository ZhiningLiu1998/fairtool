"""
FairDataset class for handling fair dataset creation and manipulation.
"""

# TODO: complement documentation, implement KFold split method

LOCAL_DEBUG = True

if not LOCAL_DEBUG:
    from ._transformer import FairDataTransformer
    from ._utils import (
        process_missing_values,
        get_group_stats,
        train_val_test_split,
        dict_values_to_percentage,
    )
    from ._check import check_feature_target_sensitive_names
    from ..utils._logging import center_str
else:  # pragma: no cover
    # For local debugging purposes
    import sys

    sys.path.append("..")
    from dataset._transformer import FairDataTransformer
    from dataset._utils import (
        process_missing_values,
        get_group_stats,
        train_val_test_split,
        dict_values_to_percentage,
    )
    from dataset._check import check_feature_target_sensitive_names
    from utils._logging import center_str

import sklearn


class FairDataset(sklearn.utils.Bunch):

    def __init__(
        self,
        data,
        target_name,
        sensitive_name,
        *,
        feature_names=None,
        categorical_features=None,
        exclude_sensitive_feature=False,
        target_pos_value=None,
        sensitive_pos_value=None,
        missing_value_handling="drop",
        missing_value_imputer=None,
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

        # missing value handling
        data = process_missing_values(
            data, how=missing_value_handling, imputer=missing_value_imputer
        )

        # transform data
        self.transformer = FairDataTransformer()
        X_, y_, s_ = self.transformer.fit_transform_from_single_dataframe(
            data=data,
            target_name=target_name,
            sensitive_name=sensitive_name,
            feature_names=feature_names,
            categorical_features=categorical_features,
            exclude_sensitive_feature=exclude_sensitive_feature,
            target_pos_value=target_pos_value,
            sensitive_pos_value=sensitive_pos_value,
            verbose=verbose,
        )

        self.X_ = X_
        self.y_ = y_
        self.s_ = s_
        self.n_samples_ = X_.shape[0]
        self.n_features_in_ = len(feature_names)
        self.n_features_out_ = X_.shape[1]
        self.feature_names_in_ = self.transformer.feature_names_in_
        self.feature_names_out_ = self.transformer.feature_names_out_
        self.feature_names_in_type_dict_ = self.transformer.feature_names_in_type_dict_
        self.feature_names_out_type_dict_ = (
            self.transformer.feature_names_out_type_dict_
        )
        self.target_name_ = self.transformer.target_name_
        self.sensitive_name_ = self.transformer.sensitive_name_
        self.data_raw_ = self.transformer.data_raw_

    def __repr__(self):
        return (
            f"FairDataset("
            f"n_samples={self.n_samples_}, n_feats_in={self.n_features_in_}, n_feats_out={self.n_features_out_}, "
            f"target={self.target_name_}, sensitive={self.sensitive_name_})"
        )

    @property
    def X_raw(self):
        return self.data_raw_[self.feature_names_in_]

    @property
    def y_raw(self):
        return self.data_raw_[self.target_name_]

    @property
    def s_raw(self):
        return self.data_raw_[self.sensitive_name_]

    @property
    def X(self):  # alias for data
        assert hasattr(
            self, "X_"
        ), "Call `_encode_features` before accessing the encoded data attribute"
        return self.X_

    @property
    def y(self):  # alias for target
        assert hasattr(
            self, "y_"
        ), "Call `_get_encoded_target` before accessing the encoded target attribute"
        return self.y_

    @property
    def s(self):  # alias for sensitive attribute
        assert hasattr(
            self, "s_"
        ), "Call `_get_encoded_sensitive_attribute` before accessing the encoded sensitive attribute"
        return self.s_

    def describe(self, full_info=False):
        dtype_feat_in_ = self.transformer.feature_names_in_type_dict_
        dtype_feat_out_ = self.transformer.feature_names_out_type_dict_
        n_feat_in = self.n_features_in_
        n_feat_in_num = len(dtype_feat_in_["numerical"])
        n_feat_in_cat = n_feat_in - n_feat_in_num
        n_feat_out = self.n_features_out_
        n_feat_out_num = len(dtype_feat_out_["numerical"])
        n_feat_out_cat = n_feat_out - n_feat_out_num
        target_encode_mapping = dict(
            list(enumerate(self.transformer._target_encoder.categories_[0]))
        )
        sensitive_encode_mapping = dict(
            list(enumerate(self.transformer._sensitive_encoder.categories_[0]))
        )
        grp_stats = get_group_stats(self.y_, self.s_)
        if full_info:
            feat_info = self.describe_feature_types()
        else:
            feat_info = ""
        info = (
            f"{center_str('FairDataset', fill_char='=')}\n"
            f"# Samples:             {self.n_samples_}\n"
            f"# Features Input:      {n_feat_in:<5d} (Numerical: {n_feat_in_num} | Categorical: {n_feat_in_cat})\n"
            f"# Features Output:     {n_feat_out:<5d} (Numerical: {n_feat_out_num} | Categorical Encoded: {n_feat_out_cat})\n"
            f"Target Attribute:      {self.target_name_.upper()} (0/1: negative/positive outcome)\n"
            f"Target Mapping:        {target_encode_mapping}\n"
            f"Target Distribution:   {grp_stats['y_group_size']}\n"
            f"Sensitive Attribute:   {self.sensitive_name_.upper()} (0/1: protected/privileged group)\n"
            f"Sensitive Mapping:     {sensitive_encode_mapping}\n"
            f"Sensitive Distribution:{grp_stats['s_group_size']}\n"
            f"Group Pos Label Ratio: {dict_values_to_percentage(grp_stats['s_group_pos_ratio'])}\n"
            f"{feat_info}"
            f"{center_str('', fill_char='=', padding=0)}"
        )
        print(info)
        return

    def describe_feature_types(self):
        dtype_feat_in_ = self.transformer.feature_names_in_type_dict_
        dtype_feat_out_ = self.transformer.feature_names_out_type_dict_
        info = f"{center_str('Input Feature Types', fill_char='-', padding=1)}\n"
        for dtype, features in dtype_feat_in_.items():
            info += f"{dtype.upper()}:\n {features}\n"
        info += f"{center_str('Output Feature Types', fill_char='-', padding=1)}\n"
        for dtype, features in dtype_feat_out_.items():
            info += f"{dtype.upper()}:\n {features}\n"
        return info

    def get_train_val_test_split(
        self,
        test_ratio,
        val_ratio=0,
        stratify_subgroup=True,
        stratify=None,
        random_state=None,
    ):
        return train_val_test_split(
            self.X_,
            self.y_,
            self.s_,
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            stratify_subgroup=stratify_subgroup,
            stratify=stratify,
            random_state=random_state,
        )

    def get_KFold_split(self, n_splits=5, shuffle=True, random_state=None):
        # TODO: implement KFold split using sklearn.model_selection.StratifiedKFold
        pass
