# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests that `vx.array` recovers Vortex extension identity from pyarrow inputs."""

from __future__ import annotations

import base64
from typing import final

import pyarrow as pa
from typing_extensions import override

import vortex as vx

# Wire format: u8 unit_tag + u16 LE tz_len. Microseconds = 1, no timezone.
# See vortex-array/src/extension/datetime/timestamp.rs.
_TIMESTAMP_US_METADATA = bytes([1, 0, 0])


@final
class VortexTimestampType(pa.ExtensionType):
    """pyarrow `ExtensionType` matching Vortex's `vortex.timestamp`."""

    _unit: str

    def __init__(self, unit: str = "us"):
        # pyarrow calls `__arrow_ext_serialize__` from __init__, so `_unit` must be set first.
        self._unit = unit
        pa.ExtensionType.__init__(self, pa.int64(), "vortex.timestamp")

    @override
    def __arrow_ext_serialize__(self) -> bytes:
        unit_tag = {"ns": 0, "us": 1, "ms": 2, "s": 3}[self._unit]
        return bytes([unit_tag, 0, 0])

    @classmethod
    @override
    def __arrow_ext_deserialize__(
        cls,
        storage_type: pa.DataType,  # noqa: ARG003
        serialized: bytes,
    ) -> VortexTimestampType:
        unit_tag = serialized[0]
        unit = {0: "ns", 1: "us", 2: "ms", 3: "s"}[unit_tag]
        return cls(unit)


def test_chunked_extension_array_uses_session_for_leaf_extension_type():
    ext_type = VortexTimestampType()
    storage = pa.array([1, 2, 3], type=pa.int64())
    arrow = pa.chunked_array(
        [pa.ExtensionArray.from_storage(ext_type, storage)]  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    )
    array = vx.array(arrow)
    assert isinstance(array, vx.ChunkedArray)
    assert repr(array.dtype) == repr(vx.timestamp("us"))
    assert repr(array.chunks()[0].dtype) == repr(vx.timestamp("us"))


def test_table_uses_session_for_extension_field_metadata():
    field = pa.field("ts", pa.int64(), nullable=False).with_metadata(
        {
            b"ARROW:extension:name": b"vortex.timestamp",
            b"ARROW:extension:metadata": base64.b64encode(_TIMESTAMP_US_METADATA),
        }
    )
    table = pa.Table.from_arrays(
        [pa.array([1, 2, 3], type=pa.int64())],
        schema=pa.schema([field]),
    )
    array = vx.array(table)
    expected = vx.struct({"ts": vx.timestamp("us")})
    assert isinstance(array, vx.ChunkedArray)
    assert repr(array.dtype) == repr(expected)
    assert repr(array.chunks()[0].dtype) == repr(expected)
