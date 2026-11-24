// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::FixedSizeBinaryArray;
use crate::arrays::fixed_size_binary::FixedSizeBinaryArrayExt;

/// Whether every value of an all-valid fixed-size binary array has the same bytes.
pub(super) fn check_fixed_size_binary_constant(array: &FixedSizeBinaryArray) -> bool {
    let width = array.byte_width() as usize;
    let values = array.values();
    match values.as_slice().split_at_checked(width) {
        Some((first, rest)) => rest.chunks_exact(width.max(1)).all(|value| value == first),
        // An array with no rows is trivially constant.
        None => true,
    }
}
