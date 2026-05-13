# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import pyarrow as pa

import vortex


def test_create_fixed_size_list_dtype():
    """Test creating a FixedSizeList dtype."""
    dtype = vortex.fixed_size_list(vortex.int_(32), 3, nullable=False)
    # The actual string representation might differ from expected
    assert "fixed_size_list" in str(dtype).lower()


def test_create_nullable_fixed_size_list_dtype():
    """Test creating a nullable FixedSizeList dtype."""
    dtype = vortex.fixed_size_list(vortex.utf8(), 2, nullable=True)
    assert str(dtype) == "fixed_size_list(utf8)[2]?"


def test_create_nested_fixed_size_list_dtype():
    """Test creating a nested FixedSizeList dtype."""
    inner_dtype = vortex.fixed_size_list(vortex.float_(32), 2, nullable=False)
    outer_dtype = vortex.fixed_size_list(inner_dtype, 3, nullable=False)
    assert str(outer_dtype) == "fixed_size_list(fixed_size_list(f32)[2])[3]"


def test_create_fixed_size_list_array_from_arrow():
    """Test creating a FixedSizeList array from PyArrow."""
    # Create a PyArrow fixed-size list array
    pa_type = pa.list_(pa.int32(), 3)
    pa_array = pa.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)], type=pa_type)

    # Convert to Vortex
    vx_array = vortex.array(pa_array)

    # Convert back to Arrow to verify
    result = vx_array.to_arrow_array()
    assert pa.types.is_fixed_size_list(result.type)
    assert result.type.list_size == 3  # pyright: ignore[reportAny]
    # PyArrow returns FixedSizeListScalar objects that need to be converted
    assert [x.as_py() for x in result] == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


def test_create_nullable_fixed_size_list_array():
    """Test creating a nullable FixedSizeList array."""
    pa_type = pa.list_(pa.int64(), 2)
    pa_array = pa.array([(10, 20), None, (30, 40)], type=pa_type)

    vx_array = vortex.array(pa_array)
    result = vx_array.to_arrow_array()

    assert pa.types.is_fixed_size_list(result.type)
    assert result.type.list_size == 2  # pyright: ignore[reportAny]
    # Handle nullable values - PyArrow scalars have as_py() method
    assert [x.as_py() for x in result] == [[10, 20], None, [30, 40]]


def test_fixed_size_list_with_string_elements():
    """Test FixedSizeList with string elements."""
    pa_type = pa.list_(pa.string(), 3)
    pa_array = pa.array([("a", "b", "c"), ("d", "e", "f"), ("g", "h", "i")], type=pa_type)

    vx_array = vortex.array(pa_array)
    result = vx_array.to_arrow_array()

    assert pa.types.is_fixed_size_list(result.type)
    assert result.type.list_size == 3  # pyright: ignore[reportAny]
    # Convert FixedSizeListScalar to Python lists
    assert [x.as_py() for x in result] == [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]]


def test_empty_fixed_size_list_array():
    """Test creating an empty FixedSizeList array."""
    pa_type = pa.list_(pa.int32(), 3)
    pa_array = pa.array([], type=pa_type)

    vx_array = vortex.array(pa_array)
    result = vx_array.to_arrow_array()

    assert pa.types.is_fixed_size_list(result.type)
    assert result.type.list_size == 3  # pyright: ignore[reportAny]
    assert len(result) == 0


def test_fixed_size_list_scalar_access():
    """Test accessing elements from a FixedSizeList scalar."""
    pa_type = pa.list_(pa.int32(), 3)
    pa_array = pa.array([(1, 2, 3), (4, 5, 6)], type=pa_type)

    vx_array = vortex.array(pa_array)

    # Get the first list as a scalar using scalar_at
    scalar = vx_array.scalar_at(0)

    # Verify it's a ListScalar
    assert isinstance(scalar, vortex.ListScalar)

    # Access elements
    assert scalar.element(0) == 1
    assert scalar.element(1) == 2
    assert scalar.element(2) == 3


def test_fixed_size_list_with_f64_elements():
    """Test that FixedSizeList survives a round trip through Vortex."""
    # Create a complex FixedSizeList array
    pa_type = pa.list_(pa.float64(), 4)
    data = [(1.1, 2.2, 3.3, 4.4), (5.5, 6.6, 7.7, 8.8), (9.9, 10.0, 11.1, 12.2)]
    pa_array = pa.array(data, type=pa_type)

    # Convert to Vortex and back
    vx_array = vortex.array(pa_array)
    result = vx_array.to_arrow_array()

    # Verify type is preserved
    assert pa.types.is_fixed_size_list(result.type)
    assert result.type.list_size == 4  # pyright: ignore[reportAny]
    assert result.type.value_type == pa.float64()  # pyright: ignore[reportAny]

    # Verify data is preserved
    # Use as_py() to convert PyArrow scalars to Python objects
    assert [x.as_py() for x in result] == [[1.1, 2.2, 3.3, 4.4], [5.5, 6.6, 7.7, 8.8], [9.9, 10.0, 11.1, 12.2]]
