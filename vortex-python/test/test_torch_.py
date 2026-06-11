# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for :class:`vortex.torch_.VortexMapDataset`, the torch-DataLoader-compatible
map-style dataset."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pyarrow as pa
import pytest
from vortex.torch_ import VortexMapDataset

import vortex as vx

from .hub_server import HubState, sample_table


def test_map_dataset(tmp_path: Path) -> None:
    path = tmp_path / "points.vortex"
    vx.io.write(pa.table({"x": list(range(100)), "y": [i * i for i in range(100)]}), str(path))

    ds = VortexMapDataset(str(path))
    assert len(ds) == 100
    assert ds[3] == {"x": 3, "y": 9}
    assert ds[-1] == {"x": 99, "y": 9801}
    assert ds.__getitems__([7, 3, 7]) == [{"x": 7, "y": 49}, {"x": 3, "y": 9}, {"x": 7, "y": 49}]
    with pytest.raises(IndexError):
        ds[100]


def test_map_dataset_projection(tmp_path: Path) -> None:
    path = tmp_path / "points.vortex"
    vx.io.write(pa.table({"x": list(range(10)), "y": list(range(10))}), str(path))

    ds = VortexMapDataset(str(path), projection=["y"])
    assert ds[4] == {"y": 4}
    assert ds.__getitems__([1, 0]) == [{"y": 1}, {"y": 0}]


def test_map_dataset_over_hf_url(local_hub: HubState) -> None:
    table = sample_table(num_rows=500)
    local_hub.publish("datasets/test-org/test-repo/resolve/main/train.vortex", table)

    ds = VortexMapDataset("hf://datasets/test-org/test-repo/train.vortex")
    assert len(ds) == 500
    assert ds[42] == {"x": table.column("x")[42].as_py(), "s": table.column("s")[42].as_py()}
    rows = cast("list[dict[str, object]]", ds.__getitems__([499, 0, 250]))
    assert [r["x"] for r in rows] == [table.column("x")[i].as_py() for i in (499, 0, 250)]
