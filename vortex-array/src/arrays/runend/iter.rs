// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::min;

use vortex_error::vortex_panic;

use crate::dtype::IntegerPType;

/// Yields exclusive run ends relative to a logical slice.
///
/// Subtracts `offset` from each end and clamps it to `length`. One end is yielded per input run,
/// including zero-length runs at the slice boundaries, so indices still match the values child.
///
/// # Panics
///
/// Panics if `offset` or `length` cannot be represented by `E`, or an input end is before `offset`.
pub fn slice_relative_ends<E: IntegerPType>(
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
        .map(move |v| {
            if v < offset_e {
                vortex_panic!("run end {v} must be >= offset {offset}");
            }
            v - offset_e
        })
        .map(move |v| min(v, length_e))
        .map(|v| v.as_())
}
