# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Exercise real Python cuDF on the GPU; missing dependencies/devices must fail."""

import unittest

import cudf


class CudfEnvironment(unittest.TestCase):
    def test_numeric_nulls(self):
        result = cudf.Series([1, None, 3]).fillna(0) + 1
        self.assertEqual(result.to_numpy().tolist(), [2, 1, 4])

    def test_string_nulls(self):
        result = cudf.Series(["red", None, "blue"]).fillna("").str.len()
        self.assertEqual(result.to_numpy().tolist(), [3, 0, 4])


if __name__ == "__main__":
    unittest.main(verbosity=2)
