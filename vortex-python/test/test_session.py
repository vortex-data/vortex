# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from pathlib import Path
from typing import cast

import pyarrow as pa

import vortex as vx


def _int64_pylist(array: vx.Array) -> list[int | None]:
    return cast(pa.Int64Array, array.to_arrow_array()).to_pylist()


def test_global_session_array_execution() -> None:

    array = vx.array([1, 2, 3])

    assert array.scalar_at(1).as_py() == 2
    assert _int64_pylist(array) == [1, 2, 3]
    assert _int64_pylist(vx.compress(array)) == [
        1,
        2,
        3,
    ]


def test_file_dataset_and_scan_use_global_session(tmp_path: Path) -> None:
    path = tmp_path / "data.vortex"

    vx.io.write(vx.array([{"x": 1}, {"x": 2}]), str(path))

    vxf = vx.open(str(path))
    dataset = vxf.to_dataset()

    assert vxf.scan().read_all().to_arrow_table().to_pylist() == [
        {"x": 1},
        {"x": 2},
    ]
    assert dataset.to_table().to_pylist() == [{"x": 1}, {"x": 2}]
