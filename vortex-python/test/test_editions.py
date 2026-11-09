# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import pyarrow as pa
import pytest

import vortex as vx

# Readers before the `vortex.zoned` layout, for example Vortex 0.73, can read this edition.
OLD_READER_EDITION = "core2025.10.0"


def table() -> pa.Table:
    n = 50_000
    return pa.table(
        {
            "int": pa.array(range(n), type=pa.int32()),
            "bigint": pa.array([None if i % 5 == 0 else i * 1_000_003 for i in range(n)], type=pa.int64()),
            "bool": pa.array([None if i % 11 == 0 else i % 2 == 0 for i in range(n)]),
            "double": pa.array([None if i % 13 == 0 else i * 0.5 for i in range(n)]),
            "string": pa.array([None if i % 7 == 0 else f"row-{i:06}" for i in range(n)]),
            "list": pa.array([[f"item-{i}", None] for i in range(n)], type=pa.list_(pa.string())),
            "nested": pa.array([{"a": i, "b": f"n{i}"} for i in range(n)]),
        }
    )


def layout_ids(path: str) -> set[bytes]:
    with open(path, "rb") as f:
        data = f.read()
    return {name for name in (b"vortex.stats", b"vortex.zoned") if name in data}


@pytest.mark.parametrize("compact", [False, True])
def test_write_options_with_edition_writes_legacy_zone_maps(tmp_path, compact):
    path = str(tmp_path / "old.vortex")
    options = vx.io.VortexWriteOptions.compact() if compact else vx.io.VortexWriteOptions.default()

    options.with_edition(OLD_READER_EDITION).write(table(), path)

    assert layout_ids(path) == {b"vortex.stats"}
    assert vx.open(path).scan().read_all().to_arrow_table().to_pylist() == table().to_pylist()


def test_write_with_edition_writes_legacy_zone_maps(tmp_path):
    path = str(tmp_path / "old.vortex")

    vx.io.write(table(), path, edition=OLD_READER_EDITION)

    assert layout_ids(path) == {b"vortex.stats"}
    assert vx.open(path).scan().read_all().to_arrow_table().to_pylist() == table().to_pylist()


def test_write_without_edition_uses_zoned_layout(tmp_path):
    path = str(tmp_path / "new.vortex")

    vx.io.write(table(), path)

    assert b"vortex.zoned" in layout_ids(path)


def test_unknown_edition_is_rejected():
    with pytest.raises(ValueError, match="Unknown Vortex edition"):
        vx.io.VortexWriteOptions.default().with_edition("core1999.01.0")
