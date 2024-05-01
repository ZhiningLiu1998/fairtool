import numpy as np
from sklearn.utils.validation import column_or_1d


def check_type(param, param_name, typs):
    """
    Check if the parameters are of the specified types.
    """
    if not isinstance(typs, tuple):
        typs = [typs]

    for typ in typs:
        if not isinstance(typ, type):
            raise ValueError(
                f"Got invalid entry with type `{get_type_name(type(param))}` in typs, "
                f"all elements in typs should be of type `type`."
            )

    if not isinstance(param, tuple(typs)):
        typ_names = [get_type_name(typ) for typ in typs]
        raise TypeError(
            f"{param_name} should be of one of following type(s): {typ_names},"
            f" got type `{get_type_name(type(param))}`."
        )


def check_1d_array(param, param_name, accept_non_numerical=False, accept_empty=False):
    """
    Check if the parameter is a 1d array.
    """
    check_type(accept_empty, "accept_empty", bool)
    try:
        column_or_1d(param)
    except Exception as e:
        raise ValueError(
            f"{param_name} must be a 1d array, got type `{get_type_name(type(param))}`."
        )
    # check if the array is numerical
    if not accept_non_numerical:
        try:
            column_or_1d(param, dtype=np.number)
        except Exception as e:
            raise ValueError(
                f"{param_name} must be a numerical array, error encountered "
                f"when checking the array: {str(e)}"
            )
    # check if the array is empty
    if not accept_empty and len(param) == 0:
        raise ValueError(f"{param_name} must not be empty.")


def get_type_name(typ):
    s = str(typ).split("'")[1]
    return s
