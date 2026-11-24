# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import decimal

import pyarrow as pa
import pytest

import vortex


def test_primitive_array_round_trip() -> None:
    a = pa.array([0, 1, 2, 3])
    arr = vortex.array(a)
    assert arr.to_arrow_array() == a


def test_array_with_nulls() -> None:
    a = pa.array([b"123", None], type=pa.string_view())
    arr = vortex.array(a)
    assert arr.to_arrow_array() == a


def test_chunked_array_with_nulls_round_trip() -> None:
    a = pa.chunked_array([[1, None, 2], [3, None]])
    arr = vortex.array(a)
    assert isinstance(arr, vortex.ChunkedArray)
    assert arr.to_arrow_array() == a


def test_chunked_array_from_chunks_takes_across_chunks() -> None:
    a = vortex.array(pa.chunked_array([[0, 1, 2], [3, 4]]))
    b = vortex.array(pa.array([5, 6, 7]))

    chunked = vortex.ChunkedArray.from_chunks([a, b])

    # The chunks of a chunked input become chunks of the result.
    assert len(chunked.chunks()) == 3
    assert len(chunked) == 8

    taken = chunked.take(vortex.array(pa.array([7, 0, 3, 3, 5], type=pa.uint64())))
    assert taken.to_arrow_array().equals(pa.array([7, 0, 3, 3, 5]))


def test_chunked_array_from_chunks_empty() -> None:
    dtype = vortex.array(pa.array([1])).dtype

    empty = vortex.ChunkedArray.from_chunks([], dtype=dtype)
    assert len(empty) == 0
    assert empty.dtype == dtype

    with pytest.raises(ValueError):
        vortex.ChunkedArray.from_chunks([])


def test_chunked_array_from_chunks_rejects_mixed_dtypes() -> None:
    with pytest.raises(RuntimeError, match="expected type"):
        vortex.ChunkedArray.from_chunks([vortex.array(pa.array([1])), vortex.array(pa.array(["a"]))])


def test_chunked_array_from_chunks_mixed_nullability() -> None:
    non_nullable = vortex.array(pa.array([1, 2]))
    nullable = vortex.array(pa.array([3, None]))

    # A nullable chunk makes the result nullable, in any position.
    for chunks in ([non_nullable, nullable], [nullable, non_nullable]):
        chunked = vortex.ChunkedArray.from_chunks(chunks)
        assert chunked.dtype == nullable.dtype

    chunked = vortex.ChunkedArray.from_chunks([non_nullable, nullable])
    taken = chunked.take(vortex.array(pa.array([3, 0, 2], type=pa.uint64())))
    assert taken.to_arrow_array().equals(pa.array([None, 1, 3]))

    # A non-nullable chunk goes into a nullable dtype.
    chunked = vortex.ChunkedArray.from_chunks([non_nullable], dtype=nullable.dtype)
    assert chunked.dtype == nullable.dtype

    # A nullable chunk does not go into a non-nullable dtype.
    with pytest.raises(RuntimeError, match="expected type"):
        vortex.ChunkedArray.from_chunks([nullable], dtype=non_nullable.dtype)


def test_decimal_array() -> None:
    a = pa.array([decimal.Decimal("1.23"), None], type=pa.decimal128(10, 2))
    arr = vortex.array(a)
    assert isinstance(arr, vortex.DecimalArray)
    assert arr.precision == 10
    assert arr.scale == 2


def test_varbin_array_round_trip() -> None:
    a = pa.array(["a", "b", "c"], type=pa.string_view())
    arr = vortex.array(a)
    assert arr.to_arrow_array() == a


@pytest.mark.parametrize(
    ("source_type", "target_type", "values"),
    [
        (pa.string_view(), pa.string(), ["one", None, "three"]),
        (pa.binary_view(), pa.binary(), [b"one", None, b"three"]),
    ],
)
def test_varbin_offset_arrow_type(
    source_type: pa.DataType,
    target_type: pa.DataType,
    values: list[str | bytes | None],
) -> None:
    source = pa.array(values, type=source_type)
    result = vortex.array(source).to_arrow_array(arrow_type=target_type)
    assert result.type == target_type
    assert result == pa.array(values, type=target_type)


def test_varbin_array_take() -> None:
    a = vortex.array(pa.array(["a", "b", "c", "d"], type=pa.string_view()))
    assert a.take(vortex.array(pa.array([0, 2]))).to_arrow_array() == pa.array(
        ["a", "c"],
        type=pa.string_view(),
    )


def test_empty_array() -> None:
    a = pa.array([], type=pa.uint8())
    primitive = vortex.array(a)
    assert primitive.to_arrow_array() == a


def test_scalar_at_out_of_bounds() -> None:
    a = vortex.array([10, 42, 999, 1992])
    with pytest.raises(IndexError, match="Index 10 out of bounds"):
        a.scalar_at(10)


@pytest.mark.parametrize(
    "arrow_type",
    [
        pa.duration("us"),
        pa.month_day_nano_interval(),
        pa.binary(3),
    ],
)
def test_unsupported_arrow_type_raises_value_error(arrow_type: pa.DataType) -> None:
    # Regression test for https://github.com/vortex-data/vortex/issues/8346:
    # unsupported Arrow types must surface as a clean ValueError, not a PanicException.
    table = pa.table({"c0": pa.array([], type=arrow_type)})
    with pytest.raises(ValueError):
        _ = vortex.array(table)


@pytest.mark.parametrize(
    ("chunks", "arrow_type"),
    [
        ([[1, 2], [3, None]], None),
        ([["a", "b"], ["c"]], pa.string()),
        ([[{"x": 1}, {"x": 2}], [{"x": 3}]], None),
    ],
)
def test_chunked_array_combine_chunks(chunks: list[list[object]], arrow_type: pa.DataType | None) -> None:
    expected = pa.chunked_array(chunks)
    arr = vortex.array(expected)
    assert isinstance(arr, vortex.ChunkedArray)

    chunked = arr.to_arrow_array(arrow_type=arrow_type)
    assert isinstance(chunked, pa.ChunkedArray)
    assert chunked.num_chunks == len(chunks)

    combined = arr.to_arrow_array(arrow_type=arrow_type, combine_chunks=True)
    assert isinstance(combined, pa.Array)
    assert chunked.equals(expected)
    assert combined.equals(expected.combine_chunks())


def test_chunked_struct_to_arrow_table_combine_chunks() -> None:
    arr = vortex.array(pa.chunked_array([[{"x": 1}], [{"x": 2}, {"x": 3}]]))
    table = arr.to_arrow_table(combine_chunks=True)
    assert table.column("x").num_chunks == 1
    assert table.column("x").to_pylist() == [1, 2, 3]
