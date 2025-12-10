# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from __future__ import annotations

from typing import final

import numpy as np
import pyarrow as pa
import pytest
from pcodec import (  # pyright: ignore[reportMissingTypeStubs]
    ChunkConfig,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType]
)
from pcodec import (  # pyright: ignore[reportMissingTypeStubs]
    wrapped as pco,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType]
)
from typing_extensions import override

import vortex as vx


@final
class PCodecArray(vx.PyArray):
    @property
    @override
    def id(self):
        return "pcodec.v0"

    @override
    def __len__(self) -> int:
        return self._len

    @property
    @override
    def dtype(self) -> vx.DType:
        return self._dtype

    def __init__(
        self,
        length: int,
        dtype: vx.DType,
        file_header: memoryview,
        chunk_header: memoryview,
        data: memoryview,
    ):
        (fd, _bytes_read) = pco.FileDecompressor.new(file_header)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

        if dtype == vx.int_(64, nullable=True):
            dt = "i64"
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        (cd, _bytes_read) = fd.read_chunk_meta(chunk_header, dt)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

        dst = np.array([0] * length, dtype=np.int64)
        cd.read_page_into(  # pyright: ignore[reportUnknownMemberType]
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
    def encode(cls, array: pa.Array[pa.Scalar[pa.DataType]], config: ChunkConfig | None = None) -> PCodecArray:  # pyright: ignore[reportUnknownParameterType]
        assert array.null_count == 0, "Cannot compress arrays with nulls"

        config = config or ChunkConfig()  # pyright: ignore[reportUnknownVariableType]

        fc = pco.FileCompressor()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        file_header = fc.write_header()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

        cc = fc.chunk_compressor(array.to_numpy(), config)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        chunk_header = cc.write_chunk_meta()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

        data = b""
        for i, _n in enumerate(cc.n_per_page()):  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportUnknownVariableType]
            data += cc.write_page(i)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

        return PCodecArray(
            len(array),
            vx.DType.from_arrow(array.type),
            file_header,  # pyright: ignore[reportUnknownArgumentType]
            chunk_header,  # pyright: ignore[reportUnknownArgumentType]
            memoryview(data),  # pyright: ignore[reportUnknownArgumentType]
        )

    @override
    @classmethod
    def decode(cls, parts: vx.ArrayParts, ctx: vx.ArrayContext, dtype: vx.DType, len: int) -> vx.Array:
        """Decode the serialized array parts into an array."""
        assert pco
        raise NotImplementedError


@pytest.mark.skip(reason="Not implemented yet")
def test_pcodec():
    _ = PCodecArray.encode(pa.array([0, 1, 2, 3, 4]))  # pyright: ignore[reportUnknownMemberType]

    # vx.registry.register(PCodecArray)
