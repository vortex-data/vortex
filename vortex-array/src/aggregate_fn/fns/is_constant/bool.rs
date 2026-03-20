// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::BoolArray;

pub(super) fn check_bool_constant(array: &BoolArray) -> bool {
    let true_count = array.to_bit_buffer().true_count();
    true_count == array.len() || true_count == 0
}
