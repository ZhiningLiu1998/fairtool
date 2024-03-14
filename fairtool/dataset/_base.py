# Copyright (c) fairtool contributors.
# Distributed under the terms of the MIT License.

import pathlib


def _get_download_data_home():
    return pathlib.Path.home() / ".fairtool-data"
