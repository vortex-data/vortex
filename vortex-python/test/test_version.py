# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import importlib.metadata

import pytest

import vortex


def test_version_matches_metadata():
    """
    Tests that we see the correct __version__
    value exported by the package.
    """
    try:
        expected = importlib.metadata.version("vortex-data")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("vortex-data distribution metadata unavailable")
    # Ensure the exported version matches the distribution metadata.
    assert vortex.__version__ == expected
