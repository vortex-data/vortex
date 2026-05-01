# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Round-trip tests for `vx.array` against pyarrow inputs carrying Vortex
extension identity over the Arrow C ABI."""

from __future__ import annotations

import base64

import pyarrow as pa

import vortex as vx

# vortex.timestamp wire format: u8 unit_tag + u16 LE tz_len (us=1, no tz).
# See vortex-array/src/extension/datetime/timestamp.rs::serialize_metadata.
_TIMESTAMP_US_METADATA = bytes([1, 0, 0])


class VortexTimestampType(pa.ExtensionType):
    """A pyarrow `ExtensionType` matching Vortex's `vortex.timestamp` extension."""

    def __init__(self, unit: str = "us"):
        # pyarrow calls `__arrow_ext_serialize__` in __init__, so set `_unit` first.
        self._unit = unit
        pa.ExtensionType.__init__(self, pa.int64(), "vortex.timestamp")

    def __arrow_ext_serialize__(self) -> bytes:
        unit_tag = {"ns": 0, "us": 1, "ms": 2, "s": 3}[self._unit]
        return bytes([unit_tag, 0, 0])

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):  # noqa: ARG003
        unit_tag = serialized[0]
        unit = {0: "ns", 1: "us", 2: "ms", 3: "s"}[unit_tag]
        return cls(unit)


def test_chunked_extension_array_uses_session_for_leaf_extension_type():
    ext_type = VortexTimestampType()
    storage = pa.array([1, 2, 3], type=pa.int64())
    arrow = pa.chunked_array([pa.ExtensionArray.from_storage(ext_type, storage)])
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
