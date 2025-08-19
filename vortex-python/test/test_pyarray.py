# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import numpy as np
import pyarrow as pa
import pytest
from pcodec import ChunkConfig  # pyright: ignore[reportAttributeAccessIssue]
from pcodec import wrapped as pco  # pyright: ignore[reportAttributeAccessIssue]

import vortex as vx


class PCodecArray(vx.PyArray):
    @property
    def id(self):
        return "pcodec.v0"

    def __len__(self) -> int:
        return self._len

    @property
    def dtype(self) -> vx.DType:
        return self._dtype

    def __init__(
        self,
        length: int,
        dtype: vx.DType,
        file_header: memoryview[bytes],
        chunk_header: memoryview[bytes],
        data: memoryview[bytes],
    ):
        (fd, _bytes_read) = pco.FileDecompressor.new(file_header)

        if dtype == vx.int_(64, nullable=True):
            dt = "i64"
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        (cd, _bytes_read) = fd.read_chunk_meta(chunk_header, dt)

        dst = np.array([0] * length, dtype=np.int64)
        cd.read_page_into(
            data,
            page_n=length,
            dst=dst,
        )

        self._len = length
        self._dtype = dtype
        self._file_header = file_header
        self._chunk_header = chunk_header
        self._data = data

    @classmethod
    def encode(cls, array: pa.Array, config: ChunkConfig | None = None) -> "PCodecArray":
        assert array.null_count == 0, "Cannot compress arrays with nulls"

        config = config or ChunkConfig()

        fc = pco.FileCompressor()
        file_header = fc.write_header()

        cc = fc.chunk_compressor(array.to_numpy(), config)
        chunk_header = cc.write_chunk_meta()

        data = b""
        for i, _n in enumerate(cc.n_per_page()):
            data += cc.write_page(i)

        return PCodecArray(
            len(array),
            vx.DType.from_arrow(array.type),
            file_header,
            chunk_header,
            memoryview(data),
        )

    @classmethod
    def decode(cls, parts: vx.ArrayParts, ctx: vx.ArrayContext, dtype: vx.DType, len: int) -> vx.Array:
        """Decode the serialized array parts into an array."""
        assert pco
        raise NotImplementedError


@pytest.mark.skip(reason="Not implemented yet")
def test_pcodec():
    PCodecArray.encode(pa.array([0, 1, 2, 3, 4]))

    vx.registry.register(PCodecArray)
