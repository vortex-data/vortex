# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import pyarrow as pa

import vortex


def test_struct() -> None:
    """
    Test struct-specific methods
    """

    # basic usage
    array = pa.Table.from_arrays([pa.array(["1", "2", "3"]), pa.array([1.0, 2.0, 3.0])], names=["strings", "floats"])
    vxarray = vortex.array(array)
    assert isinstance(vxarray, vortex.ChunkedArray)
    struct_array = vxarray.chunks()[0]
    assert isinstance(struct_array, vortex.StructArray)
    assert struct_array.names() == ["strings", "floats"]

    # advanced: duplicate field names
    array = pa.Table.from_arrays(
        [pa.array(["1", "2", "3"]), pa.array([1.0, 2.0, 3.0]), pa.array(["one", "two", "three"])],
        names=["strings", "floats", "strings"],
    )
    vxarray = vortex.array(array)
    assert isinstance(vxarray, vortex.ChunkedArray)
    struct_array = vxarray.chunks()[0]
    assert isinstance(struct_array, vortex.StructArray)
    assert struct_array.names() == ["strings", "floats", "strings"]


def test_chunked() -> None:
    chunked_array = vortex.array(
        pa.chunked_array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )
    )

    assert isinstance(chunked_array, vortex.ChunkedArray)
    chunks = chunked_array.chunks()
    assert len(chunks) == 2
    assert chunks[0].to_arrow_array() == pa.array([1.0, 2.0, 3.0])
    assert chunks[1].to_arrow_array() == pa.array([4.0, 5.0, 6.0])
