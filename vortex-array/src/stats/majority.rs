// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The strict majority of a slice: a value filling more than half of its valid positions.
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
//! compiler keep the lanes in vector registers. A second pass counts the merged candidate, since
//! the vote alone cannot tell a majority from the last survivor of an input without one.
//!
//! On x86-64 the loops are also compiled for AVX2 and chosen at runtime. The baseline target has
//! no 64-bit vector compare, which leaves 64-bit keys to scalar code in some instantiations.
//!
//! [Boyer–Moore majority vote]: https://en.wikipedia.org/wiki/Boyer%E2%80%93Moore_majority_vote_algorithm

use vortex_buffer::BitBuffer;
use vortex_mask::Mask;

use crate::dtype::NativePType;
use crate::dtype::half::f16;
use crate::dtype::i256;

/// The number of independent votes run over interleaved elements.
const LANES: usize = 16;

/// The number of strided values [`sampled_majority`] inspects before it votes.
const SAMPLE_LEN: usize = 1024;

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

/// Returns the value filling more than half of the valid positions of `values`, and its count.
///
/// `validity` must have the same length as `values`. Returns `None` when no value is a strict
/// majority, including when no position is valid.
pub fn majority<T: MajorityKey>(values: &[T], validity: &Mask) -> Option<(T, usize)> {
    // The per-lane counters are `u32`, which cannot overflow below this length.
    if values.len() > u32::MAX as usize {
        return None;
    }
    let (candidate, count) = vote_and_count(values, validity)?;
    (count > validity.true_count() / 2).then_some((candidate, count))
}

/// Like [`majority`], but returns `None` without voting when a strided sample of the valid
/// values shows no value filling more than 40% of it.
///
/// That saves two passes over inputs without a majority, for callers that fall back to an exact
/// count. The sample can miss a majority in adversarial inputs, such as values that repeat with
/// the sampling stride, so `None` does not prove that there is none.
pub fn sampled_majority<T: MajorityKey>(values: &[T], validity: &Mask) -> Option<(T, usize)> {
    if values.len() >= 4 * SAMPLE_LEN {
        let stride = values.len() / SAMPLE_LEN;
        let sample: Vec<T> = (0..SAMPLE_LEN)
            .map(|index| index * stride)
            .filter(|&index| validity.value(index))
            .map(|index| values[index])
            .collect();
        let (_, count) = vote_and_count(&sample, &Mask::new_true(sample.len()))?;
        if count * 5 <= sample.len() * 2 {
            return None;
        }
    }
    majority(values, validity)
}

/// Returns the merged candidate of a lane vote over the valid values and its exact count, or
/// `None` when no value is valid.
fn vote_and_count<T: MajorityKey>(values: &[T], validity: &Mask) -> Option<(T, usize)> {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the CPU supports AVX2, which is the only feature the kernel enables.
        return unsafe { vote_and_count_avx2(values, validity) };
    }
    vote_and_count_portable(values, validity)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn vote_and_count_avx2<T: MajorityKey>(values: &[T], validity: &Mask) -> Option<(T, usize)> {
    vote_and_count_portable(values, validity)
}

#[expect(
    clippy::inline_always,
    reason = "the loops must inline into the AVX2 kernel to be compiled for it"
)]
#[inline(always)]
fn vote_and_count_portable<T: MajorityKey>(values: &[T], validity: &Mask) -> Option<(T, usize)> {
    match validity {
        _ if validity.true_count() == 0 => None,
        Mask::AllTrue(_) => {
            let candidate = lane_vote(values);
            Some((candidate, count_equal(values, candidate)))
        }
        Mask::AllFalse(_) => None,
        Mask::Values(mask_values) => {
            let bits = mask_values.bit_buffer();
            let candidate = masked_lane_vote(values, bits);
            Some((candidate, masked_count_equal(values, bits, candidate)))
        }
    }
}

/// One step of a vote over a valid value, written without branches so the lanes vectorize.
#[expect(clippy::inline_always, reason = "see `vote_and_count_portable`")]
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
#[expect(clippy::inline_always, reason = "see `vote_and_count_portable`")]
#[inline(always)]
fn masked_vote<T: MajorityKey>(candidate: &mut T, count: &mut u32, value: T, valid: bool) {
    let empty = *count == 0;
    let agrees = candidate.key_eq(value);
    *candidate = if valid & empty { value } else { *candidate };
    let up = valid & (empty | agrees);
    let down = valid & !(empty | agrees);
    *count = *count + u32::from(up) - u32::from(down);
}

/// Merges the per-lane votes into one.
///
/// Merging two votes for different candidates pairs off the smaller surplus against the larger,
/// so a value filling more than half of the input still cannot be paired off entirely. Lanes that
/// saw no value hold a zero count, which merges as a no-op.
fn merge<T: MajorityKey>(candidates: [T; LANES], counts: [u32; LANES]) -> T {
    candidates
        .into_iter()
        .zip(counts)
        .fold(
            (T::default(), 0u32),
            |(candidate, count), (lane, lane_count)| {
                if candidate.key_eq(lane) {
                    (candidate, count + lane_count)
                } else if count >= lane_count {
                    (candidate, count - lane_count)
                } else {
                    (lane, lane_count - count)
                }
            },
        )
        .0
}

#[expect(clippy::inline_always, reason = "see `vote_and_count_portable`")]
#[inline(always)]
fn lane_vote<T: MajorityKey>(values: &[T]) -> T {
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
    merge(candidates, counts)
}

#[expect(clippy::inline_always, reason = "see `vote_and_count_portable`")]
#[inline(always)]
fn masked_lane_vote<T: MajorityKey>(values: &[T], validity: &BitBuffer) -> T {
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
    merge(candidates, counts)
}

#[expect(clippy::inline_always, reason = "see `vote_and_count_portable`")]
#[inline(always)]
fn count_equal<T: MajorityKey>(values: &[T], candidate: T) -> usize {
    values
        .iter()
        .map(|&value| usize::from(value.key_eq(candidate)))
        .sum()
}

#[expect(clippy::inline_always, reason = "see `vote_and_count_portable`")]
#[inline(always)]
fn masked_count_equal<T: MajorityKey>(values: &[T], validity: &BitBuffer, candidate: T) -> usize {
    values
        .chunks(64)
        .zip(validity.chunks().iter_padded())
        .map(|(chunk, bits)| {
            let equal = chunk
                .iter()
                .enumerate()
                .fold(0u64, |equal, (index, &value)| {
                    equal | (u64::from(value.key_eq(candidate)) << index)
                });
            (equal & bits).count_ones() as usize
        })
        .sum()
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
        let expected = values.iter().filter(|&&value| value == i256::MAX).count();
        assert_eq!(
            majority(&values, &Mask::new_true(len)),
            Some((i256::MAX, expected))
        );
    }

    #[test]
    fn exactly_half_is_not_a_majority() {
        let values = [1i128, 2, 1, 2];
        assert_eq!(majority(&values, &Mask::new_true(4)), None);
    }

    #[test]
    fn nothing_valid() {
        assert_eq!(majority::<u32>(&[], &Mask::new_true(0)), None);
        assert_eq!(majority(&[5u32; 10], &Mask::new_false(10)), None);
    }

    /// Zero fills most positions, but most zeros are invalid, so nine is the majority.
    #[rstest]
    #[case::short(40)]
    #[case::sampled(8_000)]
    fn invalid_values_do_not_vote(#[case] len: usize) {
        let values: Vec<u64> = (0..len)
            .map(|index| if index % 8 < 3 { 9 } else { 0 })
            .collect();
        let validity = Mask::from_iter((0..len).map(|index| index % 8 < 4));
        let expected = Some((9, len / 8 * 3));
        assert_eq!(majority(&values, &validity), expected);
        assert_eq!(sampled_majority(&values, &validity), expected);
    }
}
