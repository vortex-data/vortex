# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import logging

import pytest

import vortex as vx

logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def session() -> vx.Session:
    return vx.Session()
