def center_str(s, length=80, fill_char="=", padding=1):
    assert isinstance(s, str)
    assert isinstance(length, int)
    assert isinstance(fill_char, str)
    assert isinstance(padding, int)
    assert len(s) + 2 * padding <= length
    assert len(fill_char) == 1
    return s.center(len(s) + 2 * padding).center(length, fill_char)
