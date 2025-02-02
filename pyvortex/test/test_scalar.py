import vortex as vx


def test_scalar_factory():
    assert isinstance(vx.scalar(True), vx.BoolScalar)
    assert vx.scalar(True).as_py() is True
