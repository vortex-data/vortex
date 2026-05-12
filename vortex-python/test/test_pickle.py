# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import io
import pickle

import pyarrow as pa
import pytest

import vortex as vx


@pytest.fixture
def session():
    return vx.Session()


def assert_arrow_array_roundtrip(arr: vx.Array, restored: vx.Array, session: vx.Session):
    assert restored.to_arrow_array(session=session) == arr.to_arrow_array(session=session)


def assert_arrow_table_roundtrip(arr: vx.Array, restored: vx.Array, session: vx.Session):
    assert restored.to_arrow_table(session=session) == arr.to_arrow_table(session=session)


def pickle_roundtrip(obj: object, session: vx.Session):
    pickled = vx.dumps(obj, session=session)
    return vx.loads(pickled, session=session)


def test_stdlib_pickle_requires_explicit_session():
    arr = vx.array([1, 2, 3])

    with pytest.raises(TypeError, match="explicit session"):
        pickle.dumps(arr)


def test_pickle_simple_array(session: vx.Session):
    arr = vx.array([1, 2, 3, 4, 5])
    restored: vx.Array = pickle_roundtrip(arr, session)  # pyright: ignore[reportAssignmentType]

    assert_arrow_array_roundtrip(arr, restored, session)


def test_pickle_array_with_nulls(session: vx.Session):
    arr = vx.array([1, None, 3, None, 5])
    restored: vx.Array = pickle_roundtrip(arr, session)  # pyright: ignore[reportAssignmentType]

    assert_arrow_array_roundtrip(arr, restored, session)


def test_pickle_string_array(session: vx.Session):
    arr = vx.array(["hello", "world", "foo", "bar"])
    restored: vx.Array = pickle_roundtrip(arr, session)  # pyright: ignore[reportAssignmentType]

    assert_arrow_array_roundtrip(arr, restored, session)


def test_pickle_struct_array(session: vx.Session):
    arr = vx.array(
        [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35},
        ]
    )
    restored: vx.Array = pickle_roundtrip(arr, session)  # pyright: ignore[reportAssignmentType]

    assert_arrow_table_roundtrip(arr, restored, session)


def test_pickle_chunked_array(session: vx.Session):
    arr = vx.array(pa.chunked_array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    restored: vx.Array = pickle_roundtrip(arr, session)  # pyright: ignore[reportAssignmentType]

    assert_arrow_array_roundtrip(arr, restored, session)


def test_pickle_large_array(session: vx.Session):
    arr = vx.array(list(range(100_000)))
    restored: vx.Array = pickle_roundtrip(arr, session)  # pyright: ignore[reportAssignmentType]

    assert_arrow_array_roundtrip(arr, restored, session)


def test_pickle_empty_array(session: vx.Session):
    arr = vx.array(pa.array([], type=pa.int64()))
    restored: vx.Array = pickle_roundtrip(arr, session)  # pyright: ignore[reportAssignmentType]

    assert_arrow_array_roundtrip(arr, restored, session)


def test_pickle_preserves_dtype(session: vx.Session):
    arr = vx.array([1, 2, 3, 4, 5])
    original_dtype = arr.dtype
    restored: vx.Array = pickle_roundtrip(arr, session)  # pyright: ignore[reportAssignmentType]

    assert str(restored.dtype) == str(original_dtype)


def test_pickle_float_array(session: vx.Session):
    arr = vx.array([1.5, 2.7, 3.14, 4.0, 5.5])
    restored: vx.Array = pickle_roundtrip(arr, session)  # pyright: ignore[reportAssignmentType]

    assert_arrow_array_roundtrip(arr, restored, session)


def test_pickle_binary_array(session: vx.Session):
    arr = vx.array([b"hello", b"world", b"foo"])
    restored: vx.Array = pickle_roundtrip(arr, session)  # pyright: ignore[reportAssignmentType]

    assert_arrow_array_roundtrip(arr, restored, session)


def test_pickle_nested_object_with_array(session: vx.Session):
    arr = vx.array([1, 2, 3])
    restored = pickle_roundtrip({"name": "numbers", "array": arr, "values": [arr]}, session)

    assert restored["name"] == "numbers"  # pyright: ignore[reportIndexIssue]
    assert_arrow_array_roundtrip(arr, restored["array"], session)  # pyright: ignore[reportIndexIssue]
    assert_arrow_array_roundtrip(arr, restored["values"][0], session)  # pyright: ignore[reportIndexIssue]


def test_pickle_file_api(session: vx.Session):
    arr = vx.array([1, 2, 3])
    file = io.BytesIO()

    vx.Pickler(file, session=session).dump(arr)
    file.seek(0)
    restored: vx.Array = vx.Unpickler(file, session=session).load()  # pyright: ignore[reportAssignmentType]

    assert_arrow_array_roundtrip(arr, restored, session)
