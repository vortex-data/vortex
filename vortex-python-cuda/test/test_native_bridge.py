# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
# pyright: reportPrivateUsage=false

import pytest
import vortex_cuda

import vortex


def test_debug_array_metadata_dtype_reads_base_vortex_array():
    array = vortex.Array.from_range(range(0, 3))

    assert vortex_cuda._debug_array_metadata_dtype(array) == str(array.dtype)


def test_metadata_bridge_primitive_array():
    array = vortex.array([1, 2, 3])

    assert vortex_cuda._debug_array_metadata_dtype(array) == str(array.dtype)
    assert vortex_cuda._debug_array_metadata_display_values(array) == "[1i64, 2i64, 3i64]"


def test_metadata_bridge_nullable_array():
    array = vortex.array([1, None, 3])

    assert vortex_cuda._debug_array_metadata_dtype(array) == str(array.dtype)
    assert vortex_cuda._debug_array_metadata_display_values(array) == "[1i64, null, 3i64]"


def test_metadata_bridge_bool_array():
    array = vortex.array([True, False, True])

    assert vortex_cuda._debug_array_metadata_dtype(array) == str(array.dtype)
    assert vortex_cuda._debug_array_metadata_display_values(array) == "[true, false, true]"


def test_metadata_bridge_struct_with_children():
    import pyarrow as pa

    arrow_table = pa.table({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    struct_array = vortex.Array.from_arrow(
        pa.StructArray.from_arrays(  # pyright: ignore[reportUnknownMemberType]
            [arrow_table.column("a").combine_chunks(), arrow_table.column("b").combine_chunks()],
            names=["a", "b"],
        )
    )

    assert vortex_cuda._debug_array_metadata_dtype(struct_array) == str(struct_array.dtype)
    assert (
        vortex_cuda._debug_array_metadata_display_values(struct_array)
        == "[{a: 1i64, b: 4f64}, {a: 2i64, b: 5f64}, {a: 3i64, b: 6f64}]"
    )


def test_export_device_array_returns_capsules_or_clean_cuda_error():
    array = vortex.Array.from_range(range(0, 3))

    if not vortex_cuda.cuda_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            _ = vortex_cuda.export_device_array(array)
        return

    schema, device_array = vortex_cuda.export_device_array(array)
    assert type(schema).__name__ == "PyCapsule"
    assert type(device_array).__name__ == "PyCapsule"
