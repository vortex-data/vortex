# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import pytest
from pytest_benchmark.fixture import BenchmarkFixture  # pyright: ignore[reportMissingTypeStubs]

import vortex as vx


@pytest.mark.parametrize("protocol", [4, 5], ids=lambda p: f"p{p}")  # pyright: ignore[reportAny]
@pytest.mark.parametrize("operation", ["dumps", "loads", "roundtrip"])
@pytest.mark.benchmark(disable_gc=True)
def test_pickle(
    benchmark: BenchmarkFixture,
    array_fixture: vx.Array,
    session: vx.Session,
    protocol: int,
    operation: str,
):
    benchmark.group = f"pickle_p{protocol}"

    if operation == "dumps":
        benchmark(lambda: vx.dumps(array_fixture, session=session, protocol=protocol))
    elif operation == "loads":
        pickled_data = vx.dumps(array_fixture, session=session, protocol=protocol)
        benchmark(lambda: vx.loads(pickled_data, session=session))
    elif operation == "roundtrip":
        benchmark(lambda: vx.loads(vx.dumps(array_fixture, session=session, protocol=protocol), session=session))
