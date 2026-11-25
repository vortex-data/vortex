// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The majority candidate of a slice: the value filling more than half of its valid positions,
//! if there is one.
//!
//! A [Boyer–Moore majority vote] finds such a value with comparisons alone. It keeps one
//! candidate and a counter, increments the counter on a match, decrements it otherwise, and
//! replaces the candidate when the counter is zero. A value filling more than half of the input
//! survives as the candidate, because every decrement pairs off two different values and each
//! pair holds at most one copy of it.
//!
//! The vote is a mergeable summary: votes over disjoint parts of the input combine into a vote
//! over all of it, with the same guarantee. [`LANES`] independent votes therefore run over
//! interleaved elements, which removes the loop-carried dependency of a single vote and lets the
//! compiler keep the lanes in vector registers. Without a majority the candidate is an arbitrary
//! valid value, so callers that need to tell the cases apart count it afterwards.
//!
//! [`fearless_simd`] compiles the loops once per instruction set level, such as AVX2 or NEON,
//! and selects one for the current CPU at runtime. LLVM vectorizes each of them.
//!
//! [Boyer–Moore majority vote]: https://en.wikipedia.org/wiki/Boyer%E2%80%93Moore_majority_vote_algorithm

use fearless_simd::Level;
use fearless_simd::Simd;
use fearless_simd::dispatch;
use fearless_simd_macros::simd;
use vortex_buffer::BitBuffer;
use vortex_mask::Mask;

use crate::dtype::NativePType;
use crate::dtype::half::f16;
use crate::dtype::i256;

/// The number of independent votes run over interleaved elements.
const LANES: usize = 16;

/// The most values voted on with `u32` lane counters before the votes are merged into a `usize`
/// one. A multiple of 64, so each chunk's validity starts on a word.
const CHUNK_LEN: usize = 1 << 31;

/// A fixed-width value that a majority vote compares for equality.
///
/// The comparison must be an equivalence relation. Floats compare by bit pattern, as in
/// [`NativePType::is_eq`], so every NaN payload is its own value and `-0.0` differs from `0.0`.
pub trait MajorityKey: Copy + Default {
    /// Returns whether the two values are the same value.
    fn key_eq(self, other: Self) -> bool;
}

macro_rules! native_majority_key {
    ($($T:ty),*) => {
        $(
            impl MajorityKey for $T {
                #[inline]
                fn key_eq(self, other: Self) -> bool {
                    NativePType::is_eq(self, other)
                }
            }
        )*
    };
}

native_majority_key!(u8, u16, u32, u64, i8, i16, i32, i64, f16, f32, f64);

/// Fixed-size byte strings, such as UUIDs or the 16-byte views of variable-length strings.
impl<const N: usize> MajorityKey for [u8; N]
where
    [u8; N]: Default,
{
    #[inline]
    fn key_eq(self, other: Self) -> bool {
        self == other
    }
}

impl MajorityKey for i128 {
    #[inline]
    fn key_eq(self, other: Self) -> bool {
        self == other
    }
}

impl MajorityKey for i256 {
    #[inline]
    fn key_eq(self, other: Self) -> bool {
        self == other
    }
}

/// Runs a Boyer–Moore vote over the valid `values` and returns its candidate.
///
/// The candidate is the value filling more than half of the valid positions whenever there is
/// one. Otherwise it is some valid value; callers that need to know which case they have count it,
/// for example with an equality comparison. `validity` must have the same length as `values`.
/// Returns `None` only when no position is valid.
pub fn majority_candidate<T: MajorityKey>(values: &[T], validity: &Mask) -> Option<T> {
    dispatch!(Level::new(), simd => majority_candidate_simd(simd, values, validity))
}

/// The vote, compiled with the target features of `simd`.
///
/// The loops below are plain Rust, inlined here so that each level's instantiation compiles them
/// with its features. The x86-64 baseline has no 64-bit vector compare, so without a higher level
/// LLVM keeps 64-bit keys in scalar code.
#[simd]
fn majority_candidate_simd<S: Simd, T: MajorityKey>(
    _: S,
    values: &[T],
    validity: &Mask,
) -> Option<T> {
    let first_valid = match validity {
        Mask::AllTrue(len) => (*len > 0).then_some(0)?,
        Mask::AllFalse(_) => return None,
        Mask::Values(mask_values) => mask_values.bit_buffer().set_indices().next()?,
    };
    let mut merged = (T::default(), 0usize);
    for (index, chunk) in values.chunks(CHUNK_LEN).enumerate() {
        let start = index * CHUNK_LEN;
        let vote = match validity {
            Mask::Values(mask_values) => masked_lane_vote(
                chunk,
                &mask_values.bit_buffer().slice(start..start + chunk.len()),
            ),
            _ => lane_vote(chunk),
        };
        merged = merge_vote(merged, (vote.0, vote.1 as usize));
    }
    // A majority keeps a positive surplus. Without one, every vote may cancel out and leave the
    // fold's default, which need not occur at all, so any valid value will do.
    Some(if merged.1 > 0 {
        merged.0
    } else {
        values[first_valid]
    })
}

/// One step of a vote over a valid value, written without branches so the lanes vectorize.
#[expect(clippy::inline_always, reason = "see `majority_candidate_simd`")]
#[inline(always)]
fn vote<T: MajorityKey>(candidate: &mut T, count: &mut u32, value: T) {
    let empty = *count == 0;
    let agrees = candidate.key_eq(value);
    *candidate = if empty { value } else { *candidate };
    *count = if empty | agrees {
        *count + 1
    } else {
        *count - 1
    };
}

/// One step of a vote that skips invalid values. Arithmetic rather than selects on the count
/// vectorizes better once validity is involved.
#[expect(clippy::inline_always, reason = "see `majority_candidate_simd`")]
#[inline(always)]
fn masked_vote<T: MajorityKey>(candidate: &mut T, count: &mut u32, value: T, valid: bool) {
    let empty = *count == 0;
    let agrees = candidate.key_eq(value);
    *candidate = if valid & empty { value } else { *candidate };
    let up = valid & (empty | agrees);
    let down = valid & !(empty | agrees);
    *count = *count + u32::from(up) - u32::from(down);
}

/// Merges two votes over disjoint parts of the input into a vote over both.
///
/// Merging votes for different candidates pairs off the smaller surplus against the larger, so a
/// value filling more than half of the input still cannot be paired off entirely. A vote that saw
/// no value has a zero surplus and merges as a no-op.
#[expect(clippy::inline_always, reason = "see `majority_candidate_simd`")]
#[inline(always)]
fn merge_vote<
    T: MajorityKey,
    C: Copy + Ord + std::ops::Add<Output = C> + std::ops::Sub<Output = C>,
>(
    (candidate, count): (T, C),
    (other, other_count): (T, C),
) -> (T, C) {
    if candidate.key_eq(other) {
        (candidate, count + other_count)
    } else if count >= other_count {
        (candidate, count - other_count)
    } else {
        (other, other_count - count)
    }
}

/// Merges the per-lane votes into one.
fn merge_lanes<T: MajorityKey>(candidates: [T; LANES], counts: [u32; LANES]) -> (T, u32) {
    candidates
        .into_iter()
        .zip(counts)
        .fold((T::default(), 0), merge_vote)
}

#[expect(clippy::inline_always, reason = "see `majority_candidate_simd`")]
#[inline(always)]
fn lane_vote<T: MajorityKey>(values: &[T]) -> (T, u32) {
    let mut candidates = [T::default(); LANES];
    let mut counts = [0u32; LANES];
    let (chunks, remainder) = values.as_chunks::<LANES>();
    for chunk in chunks {
        for lane in 0..LANES {
            vote(&mut candidates[lane], &mut counts[lane], chunk[lane]);
        }
    }
    for (lane, &value) in remainder.iter().enumerate() {
        vote(&mut candidates[lane], &mut counts[lane], value);
    }
    merge_lanes(candidates, counts)
}

#[expect(clippy::inline_always, reason = "see `majority_candidate_simd`")]
#[inline(always)]
fn masked_lane_vote<T: MajorityKey>(values: &[T], validity: &BitBuffer) -> (T, u32) {
    let mut candidates = [T::default(); LANES];
    let mut counts = [0u32; LANES];
    let mut words = validity.chunks().iter_padded();
    let (chunks, remainder) = values.as_chunks::<64>();
    for (chunk, bits) in chunks.iter().zip(&mut words) {
        for block in 0..64 / LANES {
            for lane in 0..LANES {
                let index = block * LANES + lane;
                let valid = (bits >> index) & 1 == 1;
                masked_vote(
                    &mut candidates[lane],
                    &mut counts[lane],
                    chunk[index],
                    valid,
                );
            }
        }
    }
    let bits = words.next().unwrap_or(0);
    for (index, &value) in remainder.iter().enumerate() {
        let valid = (bits >> index) & 1 == 1;
        masked_vote(
            &mut candidates[index % LANES],
            &mut counts[index % LANES],
            value,
            valid,
        );
    }
    merge_lanes(candidates, counts)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::shorter_than_lanes(7)]
    #[case::not_a_multiple_of_64(1_001)]
    fn wide_keys(#[case] len: usize) {
        let values: Vec<i256> = (0..len)
            .map(|index| {
                if index % 3 == 0 {
                    i256::from_i128(index as i128)
                } else {
                    i256::MAX
                }
            })
            .collect();
        assert_eq!(
            majority_candidate(&values, &Mask::new_true(len)),
            Some(i256::MAX)
        );
    }

    #[test]
    fn fixed_size_bytes() {
        let values: Vec<[u8; 16]> = (0u8..100)
            .map(|index| {
                if index % 4 == 0 {
                    [index; 16]
                } else {
                    *b"0123456789abcdef"
                }
            })
            .collect();
        assert_eq!(
            majority_candidate(&values, &Mask::new_true(values.len())),
            Some(*b"0123456789abcdef")
        );
    }

    /// Every lane's vote cancels out, which leaves the fold's default, absent from the input.
    #[rstest]
    #[case::all_valid(Mask::new_true(32))]
    #[case::nullable(Mask::from_iter((0..32).map(|index| index % 2 == 1)))]
    fn cancelled_votes_return_a_valid_value(#[case] validity: Mask) {
        let values: Vec<u32> = (1..=32).collect();
        let candidate = majority_candidate(&values, &validity).expect("values are valid");
        let index = values.iter().position(|&value| value == candidate);
        assert!(index.is_some_and(|index| validity.value(index)));
    }

    #[test]
    fn nothing_valid() {
        assert_eq!(majority_candidate::<u32>(&[], &Mask::new_true(0)), None);
        assert_eq!(majority_candidate(&[5u32; 10], &Mask::new_false(10)), None);
    }

    /// Zero fills most positions, but most zeros are invalid, so nine is the majority.
    #[rstest]
    #[case::short(40)]
    #[case::long(8_000)]
    fn invalid_values_do_not_vote(#[case] len: usize) {
        let values: Vec<u64> = (0..len)
            .map(|index| if index % 8 < 3 { 9 } else { 0 })
            .collect();
        let validity = Mask::from_iter((0..len).map(|index| index % 8 < 4));
        assert_eq!(majority_candidate(&values, &validity), Some(9));
    }
}
