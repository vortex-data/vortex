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
    # Ray's uv_runtime_env_hook would auto-upload the working directory to
    # workers, but vortex-python's compiled _lib extension exceeds Ray's
    # 512 MiB upload limit. Disable the hook for these local-mode tests.
    # (Ray 2.55 added a string-type validation that broke the previous
    # `working_dir: None` workaround from ray-project/ray#53848.)
    import ray._private.ray_constants as ray_constants

    ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = False
    _ = ray.init()  # pyright: ignore[reportUnknownMemberType]
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

    tbl = pa.concat_tables(pa.Table.from_pydict(x) for x in ds.iter_batches())  # pyright: ignore[reportArgumentType, reportUnknownMemberType, reportUnknownVariableType]
    expected = pa.Table.from_pylist([record(x) for x in range(0, 10)], schema=tbl.schema)

    assert tbl == expected


def test_read_task_row_limit(tmpdir_factory):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    # Regression test: per_task_row_limit used to be passed through as the *batch size*
    # instead of capping the number of rows produced by the task.
    from vortex.ray.datasource import _read_task  # pyright: ignore[reportPrivateUsage]

    folder = tmpdir_factory.mktemp("data")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    p1, p2 = str(folder / "01.vortex"), str(folder / "02.vortex")  # pyright: ignore[reportUnknownArgumentType]
    vx.io.write(pa.table({"x": pa.array([1, 2, 3])}), p1)
    vx.io.write(pa.table({"x": pa.array([4, 5, 6])}), p2)

    task = _read_task([p1, p2], None, None, None, row_limit=4)
    assert sum(len(df) for df in task.read_fn()) == 4
    assert task.metadata.num_rows == 4

    unlimited = _read_task([p1, p2], None, None, None)
    assert sum(len(df) for df in unlimited.read_fn()) == 6
