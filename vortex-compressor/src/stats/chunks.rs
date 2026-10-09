// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The chunk loop shared by the integer and float statistics.

use vortex_compute::lane_kernels::for_each_mask_word;
use vortex_compute::lane_kernels::low_bits_mask;
use vortex_mask::AllOr;
use vortex_mask::Mask;

/// Calls `f` with each chunk of 64 `values` and its validity word, where bit `i` is set if
/// `chunk[i]` is valid.
///
/// A chunk shorter than 64 values, at either end, is padded into a full chunk whose padding bits
/// are unset. An all-valid mask takes a plain pass over the chunks.
///
/// # Panics
///
/// Panics if `validity` is all invalid, which the statistics handle before calling this.
#[allow(clippy::inline_always)]
#[inline(always)]
pub(super) fn for_each_chunk<T: Copy>(
    values: &[T],
    validity: &Mask,
    mut f: impl FnMut(&[T; 64], u64),
) {
    match validity.bit_buffer() {
        AllOr::All => {
            let (chunks, remainder) = values.as_chunks::<64>();
            chunks.iter().for_each(|chunk| f(chunk, u64::MAX));
            if !remainder.is_empty() {
                f(&padded(remainder), low_bits_mask(remainder.len()));
            }
        }
        AllOr::None => unreachable!("All invalid arrays have been handled before"),
        AllOr::Some(bits) => for_each_mask_word(bits, |word, start, len| {
            let values = &values[start..start + len];
            match <&[T; 64]>::try_from(values) {
                Ok(chunk) => f(chunk, word),
                Err(_) => f(&padded(values), word),
            }
        }),
    }
}

/// Pads fewer than 64 values into a chunk, repeating the first value.
fn padded<T: Copy>(values: &[T]) -> [T; 64] {
    let mut chunk = [values[0]; 64];
    chunk[..values.len()].copy_from_slice(values);
    chunk
}

/// Returns how many times the value changes along `values`, starting from `prev`, the last
/// value before them.
///
/// The count is branch-free and at most 64, so a `u8` accumulator lets the comparison use
/// full-width byte lanes.
#[allow(clippy::inline_always)]
#[inline(always)]
pub(super) fn count_transitions<T: PartialEq>(prev: &T, values: &[T]) -> u8 {
    debug_assert!(!values.is_empty() && values.len() <= 64);
    u8::from(values[0] != *prev)
        + values
            .iter()
            .zip(&values[1..])
            .map(|(a, b)| u8::from(a != b))
            .sum::<u8>()
}
