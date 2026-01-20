# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import pyarrow as pa
import pytest
import ray
from ray.data import read_datasource  # pyright: ignore[reportUnknownVariableType]

import vortex as vx
from vortex.ray.datasource import VortexDatasource, partition

from .test_file import record


@pytest.fixture(scope="module")
def ray_init():
    # https://github.com/ray-project/ray/issues/53848#issuecomment-3056271943
    ray.init(  # pyright: ignore[reportUnknownMemberType]
        runtime_env={
            "working_dir": None,
            "excludes": [".git", ".venv"],
        }
    )
    yield None
    ray.shutdown()  # pyright: ignore[reportUnknownMemberType]


def test_partition():
    assert partition(1, []) == [[]]
    assert partition(1, [1]) == [[1]]
    assert partition(1, [1, 2, 3]) == [[1, 2, 3]]

    assert partition(2, [1, 2, 3]) == [[1, 2], [3]]
    assert partition(3, [1, 2, 3]) == [[1], [2], [3]]

    assert partition(2, list(range(9))) == [[0, 1, 2, 3, 4], [5, 6, 7, 8]]
    assert partition(3, list(range(9))) == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    assert partition(3, list(range(11))) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10]]


def test_vortex_datasource(ray_init, tmpdir_factory):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType, reportUnusedParameter]
    folder = tmpdir_factory.mktemp("data")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    arr1 = vx.array([record(x) for x in range(5)])
    vx.io.write(arr1, str(folder / "01.vortex"))  # pyright: ignore[reportUnknownArgumentType]

    arr2 = vx.array([record(x) for x in range(5, 10)])
    vx.io.write(arr2, str(folder / "02.vortex"))  # pyright: ignore[reportUnknownArgumentType]

    ds = read_datasource(VortexDatasource(url=str(folder)))  # pyright: ignore[reportUnknownArgumentType]

    # Without an explicit sort, Ray may reorder rows *even within a single record batch*.
    ds = ds.sort("index")

    tbl = pa.concat_tables(pa.Table.from_pydict(x) for x in ds.iter_batches())  # pyright: ignore[reportArgumentType]
    expected = pa.Table.from_pylist([record(x) for x in range(0, 10)], schema=tbl.schema)

    assert tbl == expected
