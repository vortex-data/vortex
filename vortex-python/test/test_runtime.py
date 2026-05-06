# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import pytest

from vortex import runtime


def test_worker_threads():
    original = runtime.worker_count()
    try:
        runtime.set_worker_threads(0)
        assert runtime.worker_count() == 0

        runtime.set_worker_threads(1)
        assert runtime.worker_count() == 1
    finally:
        runtime.set_worker_threads(original)


def test_set_worker_threads_to_available_parallelism():
    original = runtime.worker_count()
    try:
        runtime.set_worker_threads_to_available_parallelism()
        assert runtime.worker_count() >= 1
    finally:
        runtime.set_worker_threads(original)


def test_set_worker_threads_rejects_negative():
    with pytest.raises(ValueError):
        runtime.set_worker_threads(-1)
