import vortex as vx


class PCodecArray(vx.PyEncoding):
    id = "pcodec"


def test_pcodec():
    vx.register(PCodecArray)
