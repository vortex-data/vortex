from pcodec import wrapped as pco

import vortex as vx


class PCodecArray(vx.PyArray):
    id = "pcodec"

    @classmethod
    def decode(cls, parts: vx.ArrayParts, ctx: vx.ArrayContext, dtype: vx.DType, len: int) -> vx.Array:
        """Decode the serialized array parts into an array."""
        assert pco


def test_pcodec():
    vx.registry.register(PCodecArray)
