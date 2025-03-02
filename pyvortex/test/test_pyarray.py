import vortex as vx


class PCodecArray(vx.PyEncoding):
    id = "pcodec"

    def decode(cls, parts: vx.ArrayParts, ctx: vx.ArrayContext, dtype: vx.DType, len: int) -> vx.Array:
        """Decode the serialized array parts into an array."""
        pass


def test_pcodec():
    vx.register(PCodecArray)
