# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import io
import pickle
from typing import cast

import pyarrow as pa

import vortex as vx


def assert_arrow_array_roundtrip(arr: vx.Array, restored: vx.Array) -> None:
    assert restored.to_arrow_array() == arr.to_arrow_array()


def assert_arrow_table_roundtrip(arr: vx.Array, restored: vx.Array) -> None:
    assert restored.to_arrow_table() == arr.to_arrow_table()


def pickle_roundtrip(obj: object) -> object:
    pickled = pickle.dumps(obj)
    return cast(object, pickle.loads(pickled))


def pickle_roundtrip_array(arr: vx.Array) -> vx.Array:
    return cast(vx.Array, pickle_roundtrip(arr))


def test_stdlib_pickle_roundtrip() -> None:
    arr = vx.array([1, 2, 3])
    restored = cast(vx.Array, cast(object, pickle.loads(pickle.dumps(arr))))

    assert_arrow_array_roundtrip(arr, restored)


def test_pickle_simple_array() -> None:
    arr = vx.array([1, 2, 3, 4, 5])
    restored = pickle_roundtrip_array(arr)

    assert_arrow_array_roundtrip(arr, restored)


def test_pickle_array_with_nulls() -> None:
    arr = vx.array([1, None, 3, None, 5])
    restored = pickle_roundtrip_array(arr)

    assert_arrow_array_roundtrip(arr, restored)


def test_pickle_string_array() -> None:
    arr = vx.array(["hello", "world", "foo", "bar"])
    restored = pickle_roundtrip_array(arr)

    assert_arrow_array_roundtrip(arr, restored)


def test_pickle_struct_array() -> None:
    arr = vx.array(
        [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35},
        ]
    )
    restored = pickle_roundtrip_array(arr)

    assert_arrow_table_roundtrip(arr, restored)


def test_pickle_chunked_array() -> None:
    arr = vx.array(pa.chunked_array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    restored = pickle_roundtrip_array(arr)

    assert_arrow_array_roundtrip(arr, restored)


def test_pickle_large_array() -> None:
    arr = vx.array(list(range(100_000)))
    restored = pickle_roundtrip_array(arr)

    assert_arrow_array_roundtrip(arr, restored)


def test_pickle_empty_array() -> None:
    arr = vx.array(pa.array([], type=pa.int64()))
    restored = pickle_roundtrip_array(arr)

    assert_arrow_array_roundtrip(arr, restored)


def test_pickle_preserves_dtype() -> None:
    arr = vx.array([1, 2, 3, 4, 5])
    original_dtype = arr.dtype
    restored = pickle_roundtrip_array(arr)

    assert str(restored.dtype) == str(original_dtype)


def test_pickle_float_array() -> None:
    arr = vx.array([1.5, 2.7, 3.14, 4.0, 5.5])
    restored = pickle_roundtrip_array(arr)

    assert_arrow_array_roundtrip(arr, restored)


def test_pickle_binary_array() -> None:
    arr = vx.array([b"hello", b"world", b"foo"])
    restored = pickle_roundtrip_array(arr)

    assert_arrow_array_roundtrip(arr, restored)


def test_pickle_nested_object_with_array() -> None:
    arr = vx.array([1, 2, 3])
    restored = cast(dict[str, object], pickle_roundtrip({"name": "numbers", "array": arr, "values": [arr]}))

    assert restored["name"] == "numbers"
    assert_arrow_array_roundtrip(arr, cast(vx.Array, restored["array"]))
    values = cast(list[object], restored["values"])
    assert_arrow_array_roundtrip(arr, cast(vx.Array, values[0]))


def test_pickle_file_api() -> None:
    arr = vx.array([1, 2, 3])
    file = io.BytesIO()

    pickle.Pickler(file).dump(arr)
    _ = file.seek(0)
    restored = cast(vx.Array, pickle.Unpickler(file).load())

    assert_arrow_array_roundtrip(arr, restored)
