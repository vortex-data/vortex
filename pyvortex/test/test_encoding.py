import pytest

import vortex as vx
from vortex.encoding import BoolArray


def test_bool_array():
    arr = vx.array([True, False, True, None])
    assert arr.dtype == vx.bool_(nullable=True)

    # Downcast to a BoolArray
    # TODO(ngates): I think this should be automatic if we have a registered Python class for the encoding
    arr = BoolArray(arr)
    assert str(arr) == "vortex.bool(0x02)(bool?, len=4)"

    # Test the bool-specific true_count method
    assert arr.true_count() == 2

    # Fail to downcast a non-boolean array BoolArray
    with pytest.raises(ValueError):
        BoolArray(vx.array([1, 2, 3]))
