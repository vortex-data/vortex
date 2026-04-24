# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import pickle

import pytest
from pytest_benchmark.fixture import BenchmarkFixture  # pyright: ignore[reportMissingTypeStubs]

import vortex as vx


@pytest.mark.parametrize("protocol", [4, 5], ids=lambda p: f"p{p}")  # pyright: ignore[reportAny]
@pytest.mark.parametrize("operation", ["dumps", "loads", "roundtrip"])
@pytest.mark.benchmark(disable_gc=True)
def test_pickle(
    benchmark: BenchmarkFixture,
    array_fixture: vx.Array,
    protocol: int,
    operation: str,
):
    benchmark.group = f"pickle_p{protocol}"

    if operation == "dumps":
        benchmark(lambda: pickle.dumps(array_fixture, protocol=protocol))
    elif operation == "loads":
        pickled_data = pickle.dumps(array_fixture, protocol=protocol)
        benchmark(lambda: pickle.loads(pickled_data))  # pyright: ignore[reportAny]
    elif operation == "roundtrip":
        benchmark(lambda: pickle.loads(pickle.dumps(array_fixture, protocol=protocol)))  # pyright: ignore[reportAny]
