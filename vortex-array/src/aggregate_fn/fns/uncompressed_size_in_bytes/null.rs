// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::NullArray;

pub(super) fn null_uncompressed_size_in_bytes(_array: &NullArray) -> u64 {
    0
}
