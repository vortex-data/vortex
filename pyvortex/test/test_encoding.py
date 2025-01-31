import pytest

import vortex as vx


def test_bool_array():
    arr = vx.array([True, False, True])
    assert arr.dtype == vx.bool_()

    # Downcast to a BoolArray
    arr = vx.encoding.BoolArray(arr)
    # TODO(ngates): this is a horrible display format!
    assert str(arr) == "vortex.bool(0x02)(bool, len=3)"

    assert arr.true_count() == 2

    # Fail to downcast to a BoolArray
    with pytest.raises(ValueError):
        vx.encoding.BoolArray(vx.array([1, 2, 3]))
