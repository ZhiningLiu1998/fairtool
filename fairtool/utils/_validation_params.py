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

def get_type_name(typ):
    s = str(typ).split("'")[1]
    return s