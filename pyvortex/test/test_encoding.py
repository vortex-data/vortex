import vortex as vx


def test_bool_array():
    arr = vx.array([True, False, True])
    assert arr.dtype == vx.bool_()

    # Downcast to a BoolArray
    arr = vx.encoding.BoolArray(arr)
    assert str(arr) == "BoolArray([True, False, True])"
