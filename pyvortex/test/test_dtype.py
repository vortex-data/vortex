import vortex as vx


def test_factories():
    assert str(vx.null()) == "null"
    assert str(vx.bool_()) == "bool"
    assert str(vx.bool_(nullable=True)) == "bool?"
    assert str(vx.int_()) == "i64"
    assert str(vx.int_(32)) == "i32"
    assert str(vx.int_(32, nullable=True)) == "i32?"
    assert str(vx.uint(32)) == "u32"
    assert str(vx.float_(16)) == "f16"
    assert str(vx.struct({"a": vx.int_(nullable=True)})) == "{a=i64?}"
