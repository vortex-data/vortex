// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::FixedSizeBinaryArray;
use crate::arrays::fixed_size_binary::FixedSizeBinaryArrayExt;

/// Compares the values of two fixed-size binary arrays whose rows are all valid.
pub(super) fn check_fixed_size_binary_identical(
    lhs: &FixedSizeBinaryArray,
    rhs: &FixedSizeBinaryArray,
) -> bool {
    lhs.byte_width() == rhs.byte_width()
        && lhs.len() == rhs.len()
        && lhs.values().as_slice() == rhs.values().as_slice()
}
