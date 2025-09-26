// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::min;

use vortex_dtype::IntegerPType;
use vortex_error::vortex_panic;

#[inline]
pub fn trimmed_ends_iter<E: IntegerPType>(
    run_ends: &[E],
    offset: usize,
    length: usize,
) -> impl Iterator<Item = usize> + use<'_, E> {
    let offset_e = E::from_usize(offset).unwrap_or_else(|| {
        vortex_panic!(
            "offset {} cannot be converted to {}",
            offset,
            std::any::type_name::<E>()
        )
    });
    let length_e = E::from_usize(length).unwrap_or_else(|| {
        vortex_panic!(
            "length {} cannot be converted to {}",
            length,
            std::any::type_name::<E>()
        )
    });
    run_ends
        .iter()
        .copied()
        .map(move |v| v - offset_e)
        .map(move |v| min(v, length_e))
        .map(|v| v.as_())
}
