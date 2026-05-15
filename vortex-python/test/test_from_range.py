# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import vortex as vx


def values(arr: vx.Array) -> list[object]:
    return [arr.scalar_at(i).as_py() for i in range(len(arr))]


def test_from_range_0_10_1():
    arr = vx.array(range(0, 10))
    assert values(arr) == list(range(0, 10))


def test_from_range_0_10_5():
    arr = vx.array(range(0, 10, 5))
    assert values(arr) == list(range(0, 10, 5))


def test_from_range_0_10_10():
    arr = vx.array(range(0, 10, 10))
    assert values(arr) == [0]


def test_from_range_0_10_100():
    arr = vx.array(range(0, 10, 100))
    assert values(arr) == [0]


def test_from_range_minus_5_5_1():
    arr = vx.array(range(-5, 5))
    assert values(arr) == list(range(-5, 5))


def test_from_range_minus_5_5_3():
    arr = vx.array(range(-5, 5, 3))
    assert values(arr) == [-5, -2, 1, 4]


def test_from_range_minus_7_minus_5():
    arr = vx.array(range(-7, -5))
    assert values(arr) == [-7, -6]


def test_from_range_invalid():
    arr = vx.array(range(10, 3))
    assert values(arr) == []

    arr = vx.array(range(0, 10, -1))
    assert values(arr) == []
