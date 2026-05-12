# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import io
import pickle
from typing import cast

import pyarrow as pa
import pytest

import vortex as vx


@pytest.fixture
def session() -> vx.Session:
    return vx.Session()


def assert_arrow_array_roundtrip(arr: vx.Array, restored: vx.Array, session: vx.Session) -> None:
    assert restored.to_arrow_array(session=session) == arr.to_arrow_array(session=session)


def assert_arrow_table_roundtrip(arr: vx.Array, restored: vx.Array, session: vx.Session) -> None:
    assert restored.to_arrow_table(session=session) == arr.to_arrow_table(session=session)


def pickle_roundtrip(obj: object, session: vx.Session) -> object:
    pickled = vx.dumps(obj, session=session)
    return vx.loads(pickled, session=session)


def pickle_roundtrip_array(arr: vx.Array, session: vx.Session) -> vx.Array:
    return cast(vx.Array, pickle_roundtrip(arr, session))


def test_stdlib_pickle_requires_explicit_session() -> None:
    arr = vx.array([1, 2, 3])

    with pytest.raises(TypeError, match="explicit session"):
        _ = pickle.dumps(arr)


def test_pickle_simple_array(session: vx.Session) -> None:
    arr = vx.array([1, 2, 3, 4, 5])
    restored = pickle_roundtrip_array(arr, session)

    assert_arrow_array_roundtrip(arr, restored, session)


def test_pickle_array_with_nulls(session: vx.Session) -> None:
    arr = vx.array([1, None, 3, None, 5])
    restored = pickle_roundtrip_array(arr, session)

    assert_arrow_array_roundtrip(arr, restored, session)


def test_pickle_string_array(session: vx.Session) -> None:
    arr = vx.array(["hello", "world", "foo", "bar"])
    restored = pickle_roundtrip_array(arr, session)

    assert_arrow_array_roundtrip(arr, restored, session)


def test_pickle_struct_array(session: vx.Session) -> None:
    arr = vx.array(
        [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35},
        ]
    )
    restored = pickle_roundtrip_array(arr, session)

    assert_arrow_table_roundtrip(arr, restored, session)


def test_pickle_chunked_array(session: vx.Session) -> None:
    arr = vx.array(pa.chunked_array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    restored = pickle_roundtrip_array(arr, session)

    assert_arrow_array_roundtrip(arr, restored, session)


def test_pickle_large_array(session: vx.Session) -> None:
    arr = vx.array(list(range(100_000)))
    restored = pickle_roundtrip_array(arr, session)

    assert_arrow_array_roundtrip(arr, restored, session)


def test_pickle_empty_array(session: vx.Session) -> None:
    arr = vx.array(pa.array([], type=pa.int64()))
    restored = pickle_roundtrip_array(arr, session)

    assert_arrow_array_roundtrip(arr, restored, session)


def test_pickle_preserves_dtype(session: vx.Session) -> None:
    arr = vx.array([1, 2, 3, 4, 5])
    original_dtype = arr.dtype
    restored = pickle_roundtrip_array(arr, session)

    assert str(restored.dtype) == str(original_dtype)


def test_pickle_float_array(session: vx.Session) -> None:
    arr = vx.array([1.5, 2.7, 3.14, 4.0, 5.5])
    restored = pickle_roundtrip_array(arr, session)

    assert_arrow_array_roundtrip(arr, restored, session)


def test_pickle_binary_array(session: vx.Session) -> None:
    arr = vx.array([b"hello", b"world", b"foo"])
    restored = pickle_roundtrip_array(arr, session)

    assert_arrow_array_roundtrip(arr, restored, session)


def test_pickle_nested_object_with_array(session: vx.Session) -> None:
    arr = vx.array([1, 2, 3])
    restored = cast(dict[str, object], pickle_roundtrip({"name": "numbers", "array": arr, "values": [arr]}, session))

    assert restored["name"] == "numbers"
    assert_arrow_array_roundtrip(arr, cast(vx.Array, restored["array"]), session)
    values = cast(list[object], restored["values"])
    assert_arrow_array_roundtrip(arr, cast(vx.Array, values[0]), session)


def test_pickle_file_api(session: vx.Session) -> None:
    arr = vx.array([1, 2, 3])
    file = io.BytesIO()

    vx.Pickler(file, session=session).dump(arr)
    _ = file.seek(0)
    restored = cast(vx.Array, vx.Unpickler(file, session=session).load())

    assert_arrow_array_roundtrip(arr, restored, session)
