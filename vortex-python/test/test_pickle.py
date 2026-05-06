# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import pickle

import pytest

import vortex as vx


def test_pickle_requires_explicit_session():
    arr = vx.array([1, 2, 3])

    with pytest.raises(TypeError, match="explicit session"):
        pickle.dumps(arr)
