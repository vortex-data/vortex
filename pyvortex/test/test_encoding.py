import pytest

import vortex as vx


def test_bool_array():
    arr = vx.array([True, False, True, None])
    assert arr.dtype == vx.bool_(nullable=True)

    # Downcast to a BoolArray
    arr = vx.encoding.BoolArray(arr)
    assert str(arr) == "vortex.bool(0x02)(bool?, len=4)"

    # Test the bool-specific true_count method
    assert arr.true_count() == 2

    # Fail to downcast a non-boolean array BoolArray
    with pytest.raises(ValueError):
        vx.encoding.BoolArray(vx.array([1, 2, 3]))
