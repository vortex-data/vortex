// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The chunk loop of the float statistics.

use vortex_array::dtype::NativePType;
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
/// Panics if `validity` is all invalid, which the caller handles before calling this.
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
/// value before them, using bitwise [`NativePType::is_eq`] equality.
///
/// Expects `1 <= values.len() <= 64`.
#[allow(clippy::inline_always)]
#[inline(always)]
pub(super) fn count_transitions<T: NativePType>(prev: T, values: &[T]) -> u32 {
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2"), not(miri)))]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: `avx2` was just checked via `is_x86_feature_detected!`.
        return unsafe { count_transitions_avx2(prev, values) };
    }
    count_transitions_inner(prev, values)
}

/// AVX2-specialized wrapper around [`count_transitions_inner`] so `f64` bitwise comparisons use
/// `vpcmpeqq` on default x86-64 builds.
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2"), not(miri)))]
#[target_feature(enable = "avx2")]
fn count_transitions_avx2<T: NativePType>(prev: T, values: &[T]) -> u32 {
    count_transitions_inner(prev, values)
}

/// Branch-free bitwise transition counter over a 1..=64 element chunk.
#[allow(clippy::inline_always)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "chunk length is bounded by 64"
)]
#[inline(always)]
fn count_transitions_inner<T: NativePType>(prev: T, values: &[T]) -> u32 {
    debug_assert!(!values.is_empty() && values.len() <= 64);
    let equal_pairs: u64 = values
        .iter()
        .zip(&values[1..])
        .map(|(&a, &b)| u64::from(a.is_eq(b)))
        .sum();
    u32::from(!values[0].is_eq(prev)) + (values.len() as u32 - 1 - equal_pairs as u32)
}
