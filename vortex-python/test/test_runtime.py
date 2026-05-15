# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import pytest

import vortex as vx


def test_worker_threads():
    original = vx.worker_threads()
    try:
        vx.set_worker_threads(0)
        assert vx.worker_threads() == 0

        vx.set_worker_threads(1)
        assert vx.worker_threads() == 1
    finally:
        vx.set_worker_threads(original)


def test_set_worker_threads_none_resets_to_available_parallelism():
    original = vx.worker_threads()
    try:
        vx.set_worker_threads(0)
        vx.set_worker_threads(None)
        assert vx.worker_threads() >= 1
    finally:
        vx.set_worker_threads(original)


def test_set_worker_threads_rejects_negative():
    with pytest.raises(ValueError):
        vx.set_worker_threads(-1)
