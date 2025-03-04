import numpy as np
import pyarrow as pa
from pcodec import ChunkConfig
from pcodec import wrapped as pco

import vortex as vx


class PCodecArray(vx.PyArray):
    id = "pcodec.v0"

    def __init__(
        self,
        length: int,
        dtype: vx.DType,
        file_header: memoryview,
        chunk_header: memoryview,
        data: memoryview,
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

        self._len = len
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

        PCodecArray(
            len(array),
            vx.DType.from_arrow(array.type),
            file_header,
            chunk_header,
            data,
        )

    @classmethod
    def decode(cls, parts: vx.ArrayParts, ctx: vx.ArrayContext, dtype: vx.DType, len: int) -> vx.Array:
        """Decode the serialized array parts into an array."""
        assert pco


def test_pcodec():
    PCodecArray.encode(pa.array([0, 1, 2, 3, 4]))

    vx.registry.register(PCodecArray)
