# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import pickle

import pyarrow as pa

import vortex as vx


def test_pickle_simple_array():
    arr = vx.array([1, 2, 3, 4, 5])
    pickled = pickle.dumps(arr)
    restored: vx.Array = pickle.loads(pickled)  # pyright: ignore[reportAny]

    assert restored.to_arrow_array() == arr.to_arrow_array()


def test_pickle_array_with_nulls():
    arr = vx.array([1, None, 3, None, 5])
    pickled = pickle.dumps(arr)
    restored: vx.Array = pickle.loads(pickled)  # pyright: ignore[reportAny]

    assert restored.to_arrow_array() == arr.to_arrow_array()


def test_pickle_string_array():
    arr = vx.array(["hello", "world", "foo", "bar"])
    pickled = pickle.dumps(arr)
    restored: vx.Array = pickle.loads(pickled)  # pyright: ignore[reportAny]

    assert restored.to_arrow_array() == arr.to_arrow_array()


def test_pickle_struct_array():
    arr = vx.array(
        [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35},
        ]
    )
    pickled = pickle.dumps(arr)
    restored: vx.Array = pickle.loads(pickled)  # pyright: ignore[reportAny]

    assert restored.to_arrow_table() == arr.to_arrow_table()


def test_pickle_chunked_array():
    arr = vx.array(pa.chunked_array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    pickled = pickle.dumps(arr)
    restored: vx.Array = pickle.loads(pickled)  # pyright: ignore[reportAny]

    assert restored.to_arrow_array() == arr.to_arrow_array()


def test_pickle_large_array():
    arr = vx.array(list(range(100_000)))
    pickled = pickle.dumps(arr)
    restored: vx.Array = pickle.loads(pickled)  # pyright: ignore[reportAny]

    assert restored.to_arrow_array() == arr.to_arrow_array()


def test_pickle_empty_array():
    arr = vx.array(pa.array([], type=pa.int64()))
    pickled = pickle.dumps(arr)
    restored: vx.Array = pickle.loads(pickled)  # pyright: ignore[reportAny]

    assert restored.to_arrow_array() == arr.to_arrow_array()


def test_pickle_different_protocols():
    arr = vx.array([1, 2, 3, 4, 5])

    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        pickled = pickle.dumps(arr, protocol=protocol)
        restored: vx.Array = pickle.loads(pickled)  # pyright: ignore[reportAny]
        assert restored.to_arrow_array() == arr.to_arrow_array()


def test_pickle_preserves_dtype():
    arr = vx.array([1, 2, 3, 4, 5])
    original_dtype = arr.dtype

    pickled = pickle.dumps(arr)
    restored: vx.Array = pickle.loads(pickled)  # pyright: ignore[reportAny]

    assert str(restored.dtype) == str(original_dtype)


def test_pickle_float_array():
    arr = vx.array([1.5, 2.7, 3.14, 4.0, 5.5])
    pickled = pickle.dumps(arr)
    restored: vx.Array = pickle.loads(pickled)  # pyright: ignore[reportAny]

    assert restored.to_arrow_array() == arr.to_arrow_array()


def test_pickle_binary_array():
    arr = vx.array([b"hello", b"world", b"foo"])
    pickled = pickle.dumps(arr)
    restored: vx.Array = pickle.loads(pickled)  # pyright: ignore[reportAny]

    assert restored.to_arrow_array() == arr.to_arrow_array()


def test_pickle_protocol_5_simple():
    arr = vx.array([1, 2, 3, 4, 5])
    pickled = pickle.dumps(arr, protocol=5)
    restored: vx.Array = pickle.loads(pickled)  # pyright: ignore[reportAny]

    assert restored.to_arrow_array() == arr.to_arrow_array()


def test_pickle_protocol_5_with_nulls():
    arr = vx.array([1, None, 3, None, 5])
    pickled = pickle.dumps(arr, protocol=5)
    restored: vx.Array = pickle.loads(pickled)  # pyright: ignore[reportAny]

    assert restored.to_arrow_array() == arr.to_arrow_array()


def test_pickle_protocol_5_large_array():
    arr = vx.array(list(range(1_000_000)))
    pickled = pickle.dumps(arr, protocol=5)
    restored: vx.Array = pickle.loads(pickled)  # pyright: ignore[reportAny]

    assert restored.to_arrow_array() == arr.to_arrow_array()


def test_pickle_protocol_5_string_array():
    arr = vx.array(["hello", "world", "protocol", "five"])
    pickled = pickle.dumps(arr, protocol=5)
    restored: vx.Array = pickle.loads(pickled)  # pyright: ignore[reportAny]

    assert restored.to_arrow_array() == arr.to_arrow_array()


def test_pickle_protocol_5_struct_array():
    arr = vx.array(
        [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35},
        ]
    )
    pickled = pickle.dumps(arr, protocol=5)
    restored: vx.Array = pickle.loads(pickled)  # pyright: ignore[reportAny]

    assert restored.to_arrow_table() == arr.to_arrow_table()


def test_pickle_protocol_comparison():
    arr = vx.array(list(range(10_000)))

    pickled_p4 = pickle.dumps(arr, protocol=4)
    pickled_p5 = pickle.dumps(arr, protocol=5)

    restored_p4: vx.Array = pickle.loads(pickled_p4)  # pyright: ignore[reportAny]
    restored_p5: vx.Array = pickle.loads(pickled_p5)  # pyright: ignore[reportAny]

    assert restored_p4.to_arrow_array() == arr.to_arrow_array()
    assert restored_p5.to_arrow_array() == arr.to_arrow_array()
    assert restored_p4.to_arrow_array() == restored_p5.to_arrow_array()


def test_pickle_protocol_5_preserves_dtype():
    arr = vx.array([1.5, 2.7, 3.14])
    original_dtype = arr.dtype

    pickled = pickle.dumps(arr, protocol=5)
    restored: vx.Array = pickle.loads(pickled)  # pyright: ignore[reportAny]

    assert str(restored.dtype) == str(original_dtype)


def test_pickle_protocol_5_chunked_array():
    arr = vx.array(pa.chunked_array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    pickled = pickle.dumps(arr, protocol=5)
    restored: vx.Array = pickle.loads(pickled)  # pyright: ignore[reportAny]

    assert restored.to_arrow_array() == arr.to_arrow_array()
