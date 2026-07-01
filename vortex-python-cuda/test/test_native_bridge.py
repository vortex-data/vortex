# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
# pyright: reportPrivateUsage=false

import gc

import pytest
import vortex_cuda

import vortex


def _require_cuda() -> None:
    if not vortex_cuda.cuda_available():
        pytest.skip("CUDA device is not available")


def _assert_exported_device_array(
    array: object, *, length: int, null_count: int, n_children: int
) -> tuple[object, object]:
    schema, device_array = vortex_cuda.export_device_array(array)
    summary = vortex_cuda._debug_arrow_device_array_capsule_summary(schema, device_array)

    assert summary["schema_live"] is True
    assert summary["array_live"] is True
    assert summary["is_cuda"] is True
    assert summary["length"] == length
    assert summary["null_count"] == null_count
    assert summary["n_children"] == n_children
    n_buffers = summary["n_buffers"]
    assert isinstance(n_buffers, int)
    assert n_buffers >= 0

    return schema, device_array


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


def test_arrow_device_export_primitive_array():
    _require_cuda()

    _ = _assert_exported_device_array(vortex.array([1, 2, 3]), length=3, null_count=0, n_children=0)


def test_arrow_device_export_nullable_primitive_array():
    _require_cuda()

    _ = _assert_exported_device_array(vortex.array([1, None, 3]), length=3, null_count=1, n_children=0)


def test_arrow_device_export_nullable_bool_array():
    _require_cuda()

    _ = _assert_exported_device_array(vortex.array([True, None, False]), length=3, null_count=1, n_children=0)


def test_arrow_device_export_string_array():
    _require_cuda()

    _ = _assert_exported_device_array(
        vortex.array(["alpha", "beta", "a longer string that should use the varbin data buffer"]),
        length=3,
        null_count=0,
        n_children=0,
    )


def test_arrow_device_export_struct_array():
    import pyarrow as pa

    _require_cuda()

    arrow_table = pa.table({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    struct_array = vortex.Array.from_arrow(
        pa.StructArray.from_arrays(  # pyright: ignore[reportUnknownMemberType]
            [arrow_table.column("a").combine_chunks(), arrow_table.column("b").combine_chunks()],
            names=["a", "b"],
        )
    )

    _ = _assert_exported_device_array(struct_array, length=3, null_count=0, n_children=2)


def test_arrow_device_capsules_drop_unconsumed():
    _require_cuda()

    schema, device_array = _assert_exported_device_array(vortex.array([1, 2, 3]), length=3, null_count=0, n_children=0)
    del schema, device_array
    _ = gc.collect()


def test_arrow_device_capsules_consumer_release_and_used_names():
    _require_cuda()

    schema, device_array = _assert_exported_device_array(vortex.array([1, 2, 3]), length=3, null_count=0, n_children=0)
    consume_result = vortex_cuda._debug_consume_arrow_device_array_capsules(schema, device_array)
    assert consume_result == (True, True, True, True, True, True)
    del schema, device_array
    _ = gc.collect()
