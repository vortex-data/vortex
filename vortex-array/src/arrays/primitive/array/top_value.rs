// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The most common value of a primitive array.
//!
//! Most callers ask for the top value of arrays that one value dominates, such as the fill value
//! of a sparse array. For those, a [Boyer–Moore majority vote] finds the value without hashing:
//! it keeps one candidate and a counter, increments the counter on a match, decrements it
//! otherwise, and replaces the candidate when the counter is zero. Any value that fills more than
//! half of the input survives as the candidate.
//!
//! The vote is a mergeable summary: votes over disjoint parts of the input combine into a vote
//! over all of it, with the same guarantee. [`LANES`] independent votes therefore run over
//! interleaved elements, which removes the loop-carried dependency of a single vote and lets the
//! compiler keep the lanes in vector registers. A second, branchless pass counts the merged
//! candidate. When it is not a strict majority, the vote says nothing about the mode, so the
//! values are counted in a hash map instead.
//!
//! [Boyer–Moore majority vote]: https://en.wikipedia.org/wiki/Boyer%E2%80%93Moore_majority_vote_algorithm

use std::hash::Hash;

use rustc_hash::FxBuildHasher;
use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_utils::aliases::hash_map::HashMap;

use crate::ExecutionCtx;
use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::NativeValue;
use crate::dtype::NativePType;
use crate::match_each_native_ptype;
use crate::scalar::PValue;

/// The number of independent votes run over interleaved elements.
const LANES: usize = 16;

impl PrimitiveArray {
    /// Compute most common present value of this array
    pub fn top_value(&self, ctx: &mut ExecutionCtx) -> VortexResult<Option<(PValue, usize)>> {
        if self.is_empty() {
            return Ok(None);
        }

        if self.validity()?.definitely_all_null() {
            return Ok(None);
        }

        let mask = self.as_ref().validity()?.execute_mask(self.len(), ctx)?;
        if mask.all_false() {
            return Ok(None);
        }

        match_each_native_ptype!(self.ptype(), |P| {
            let (top, count) = typed_top_value(self.as_slice::<P>(), &mask);
            Ok(Some((top.into(), count)))
        })
    }
}

/// Returns the most common valid value and its count. At least one value must be valid.
fn typed_top_value<T>(values: &[T], mask: &Mask) -> (T, usize)
where
    T: NativePType,
    NativeValue<T>: Eq + Hash,
{
    if size_of::<T>() == 1 {
        return byte_top_value(values, mask);
    }

    // The per-lane vote counters are `u32`, which cannot overflow below this length.
    if values.len() > u32::MAX as usize {
        return hash_top_value(values, mask);
    }

    if !sample_has_near_majority(values, mask) {
        return hash_top_value(values, mask);
    }

    let (candidate, count) = match mask {
        Mask::AllTrue(_) => {
            let candidate = majority_candidate(values);
            (candidate, count_equal(values, candidate))
        }
        Mask::AllFalse(_) => unreachable!("all-invalid arrays are handled by the caller"),
        Mask::Values(mask_values) => {
            let validity = mask_values.bit_buffer();
            let candidate = masked_majority_candidate(values, validity);
            (candidate, masked_count_equal(values, validity, candidate))
        }
    };

    if count > mask.true_count() / 2 {
        return (candidate, count);
    }
    hash_top_value(values, mask)
}

/// The number of values [`sample_has_near_majority`] looks at.
const SAMPLE_LEN: usize = 1024;

/// Returns whether some value fills more than 40% of a strided sample of the valid values.
///
/// Without one, the input almost certainly has no majority, and the vote would only add two passes
/// before the hash map. The sample decides which exact method runs, so it never changes the result.
/// It is strided rather than a prefix so that sorted or clustered inputs are sampled evenly.
fn sample_has_near_majority<T: NativePType>(values: &[T], mask: &Mask) -> bool {
    if values.len() < 4 * SAMPLE_LEN {
        return true;
    }
    let stride = values.len() / SAMPLE_LEN;
    let sample: Vec<T> = (0..SAMPLE_LEN)
        .map(|index| index * stride)
        .filter(|&index| mask.value(index))
        .map(|index| values[index])
        .collect();
    if sample.is_empty() {
        return true;
    }
    let candidate = majority_candidate(&sample);
    count_equal(&sample, candidate) * 5 > sample.len() * 2
}

/// One step of a Boyer–Moore vote over a valid value, written without branches so the lanes
/// vectorize.
#[inline]
fn vote_valid<T: NativePType>(candidate: &mut T, count: &mut u32, value: T) {
    let empty = *count == 0;
    let agrees = candidate.is_eq(value);
    *candidate = if empty { value } else { *candidate };
    *count = if empty | agrees {
        *count + 1
    } else {
        *count - 1
    };
}

/// One step of a Boyer–Moore vote that skips invalid values. Arithmetic rather than selects on
/// the count vectorizes better once validity is involved.
#[inline]
fn vote<T: NativePType>(candidate: &mut T, count: &mut u32, value: T, valid: bool) {
    let empty = *count == 0;
    let agrees = candidate.is_eq(value);
    *candidate = if valid & empty { value } else { *candidate };
    let up = valid & (empty | agrees);
    let down = valid & !(empty | agrees);
    *count = *count + u32::from(up) - u32::from(down);
}

/// Merges the per-lane votes into one, preserving the majority guarantee.
///
/// Each lane's vote pairs off its elements into pairs of different values plus `count` copies of
/// its candidate. Merging two votes for different candidates pairs off the smaller surplus against
/// the larger, so a value filling more than half of the input still cannot be paired off entirely.
fn merge_votes<T: NativePType>(candidates: [T; LANES], counts: [u32; LANES]) -> (T, u32) {
    candidates.into_iter().zip(counts).fold(
        (T::default(), 0),
        |(candidate, count), (lane, lane_count)| {
            if candidate.is_eq(lane) {
                (candidate, count + lane_count)
            } else if count >= lane_count {
                (candidate, count - lane_count)
            } else {
                (lane, lane_count - count)
            }
        },
    )
}

/// Returns the strict majority of `values` if it has one, and an arbitrary value otherwise.
fn majority_candidate<T: NativePType>(values: &[T]) -> T {
    let mut candidates = [T::default(); LANES];
    let mut counts = [0u32; LANES];
    let (chunks, remainder) = values.as_chunks::<LANES>();
    for chunk in chunks {
        for lane in 0..LANES {
            vote_valid(&mut candidates[lane], &mut counts[lane], chunk[lane]);
        }
    }
    for (lane, &value) in remainder.iter().enumerate() {
        vote_valid(&mut candidates[lane], &mut counts[lane], value);
    }
    merge_votes(candidates, counts).0
}

/// Returns the strict majority of the valid `values` if they have one, and an arbitrary value
/// otherwise.
fn masked_majority_candidate<T: NativePType>(values: &[T], validity: &BitBuffer) -> T {
    let mut candidates = [T::default(); LANES];
    let mut counts = [0u32; LANES];
    let mut words = validity.chunks().iter_padded();
    let (chunks, remainder) = values.as_chunks::<64>();
    for (chunk, bits) in chunks.iter().zip(&mut words) {
        for block in 0..64 / LANES {
            for lane in 0..LANES {
                let index = block * LANES + lane;
                let valid = (bits >> index) & 1 == 1;
                vote(
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
        let lane = index % LANES;
        vote(
            &mut candidates[lane],
            &mut counts[lane],
            value,
            (bits >> index) & 1 == 1,
        );
    }
    merge_votes(candidates, counts).0
}

fn count_equal<T: NativePType>(values: &[T], candidate: T) -> usize {
    values
        .iter()
        .map(|&value| usize::from(value.is_eq(candidate)))
        .sum()
}

fn masked_count_equal<T: NativePType>(values: &[T], validity: &BitBuffer, candidate: T) -> usize {
    values
        .chunks(64)
        .zip(validity.chunks().iter_padded())
        .map(|(chunk, bits)| {
            let equal = chunk
                .iter()
                .enumerate()
                .fold(0u64, |equal, (index, &value)| {
                    equal | (u64::from(value.is_eq(candidate)) << index)
                });
            (equal & bits).count_ones() as usize
        })
        .sum()
}

/// Counts every valid value in a hash map and returns the most common one.
fn hash_top_value<T>(values: &[T], mask: &Mask) -> (T, usize)
where
    T: NativePType,
    NativeValue<T>: Eq + Hash,
{
    let mut distinct_values: HashMap<NativeValue<T>, usize, FxBuildHasher> =
        HashMap::with_hasher(FxBuildHasher);
    match mask.indices() {
        AllOr::All => {
            for &value in values {
                *distinct_values.entry(NativeValue(value)).or_insert(0) += 1;
            }
        }
        AllOr::None => unreachable!("all-invalid arrays are handled by the caller"),
        AllOr::Some(indices) => {
            for &index in indices {
                *distinct_values
                    .entry(NativeValue(values[index]))
                    .or_insert(0) += 1;
            }
        }
    }

    let (&top_value, &top_count) = distinct_values
        .iter()
        .max_by_key(|&(_, &count)| count)
        .vortex_expect("non-empty");
    (top_value.0, top_count)
}

/// Counts one-byte values exactly in a histogram over all 256 of them.
fn byte_top_value<T: NativePType>(values: &[T], mask: &Mask) -> (T, usize) {
    let byte = |value: T| value.to_le_bytes()[0];
    let mut counts = [0usize; 256];
    match mask.indices() {
        AllOr::All => {
            // Separate tables keep runs of one value from serializing on a single counter.
            let mut tables = [[0u32; 256]; 4];
            let (chunks, remainder) = values.as_chunks::<4>();
            for chunk in chunks {
                for (table, &value) in tables.iter_mut().zip(chunk) {
                    table[usize::from(byte(value))] += 1;
                }
            }
            for &value in remainder {
                tables[0][usize::from(byte(value))] += 1;
            }
            for table in &tables {
                for (count, &table_count) in counts.iter_mut().zip(table) {
                    *count += table_count as usize;
                }
            }
        }
        AllOr::None => unreachable!("all-invalid arrays are handled by the caller"),
        AllOr::Some(indices) => indices
            .iter()
            .for_each(|&index| counts[usize::from(byte(values[index]))] += 1),
    }
    let (top_byte, &top_count) = counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, &count)| count)
        .vortex_expect("the histogram is non-empty");
    let top_byte = u8::try_from(top_byte).vortex_expect("histogram index fits in a byte");
    let top = T::try_from_le_bytes(&[top_byte]).vortex_expect("one byte encodes a one-byte type");
    (top, top_count)
}

#[cfg(test)]
#[expect(clippy::cast_possible_truncation, reason = "test inputs are small")]
mod tests {
    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::*;
    use crate::VortexSessionExecute;
    use crate::validity::Validity;

    fn top_value(array: &PrimitiveArray) -> VortexResult<Option<(PValue, usize)>> {
        array.top_value(&mut VortexSession::empty().create_execution_ctx())
    }

    /// The majority sits at the end of a sorted array, so a prefix of the input never sees it.
    #[rstest]
    #[case::shorter_than_lanes(5)]
    #[case::not_a_multiple_of_64(1_000)]
    #[case::sampled(100_003)]
    fn sorted_majority(#[case] len: usize) -> VortexResult<()> {
        let minority = len / 3;
        let values: Buffer<u32> = (0..len)
            .map(|index| {
                if index < minority {
                    index as u32
                } else {
                    u32::MAX
                }
            })
            .collect();
        let array = PrimitiveArray::new(values, Validity::NonNullable);
        assert_eq!(top_value(&array)?, Some((u32::MAX.into(), len - minority)));
        Ok(())
    }

    /// Without a majority, the vote's candidate is arbitrary, so the mode comes from counting.
    #[rstest]
    #[case::short(30)]
    #[case::sampled(30_000)]
    fn plurality_without_majority(#[case] len: usize) -> VortexResult<()> {
        // 40% sevens, the rest split evenly between 1, 2 and 3, interleaved so that every
        // lane sees a mix.
        let values: Buffer<i64> = (0..len)
            .map(|index| match index % 10 {
                0..4 => 7,
                4 | 5 => 1,
                6 | 7 => 2,
                _ => 3,
            })
            .collect();
        let array = PrimitiveArray::new(values, Validity::NonNullable);
        assert_eq!(top_value(&array)?, Some((7i64.into(), len * 4 / 10)));
        Ok(())
    }

    /// Invalid values must not vote: zero is the most common stored value, but most zeros are null.
    #[rstest]
    #[case::short(104)]
    #[case::sampled(10_000)]
    fn nulls_do_not_vote(#[case] len: usize) -> VortexResult<()> {
        let values: Buffer<u16> = (0..len)
            .map(|index| if index % 8 < 3 { 9 } else { 0 })
            .collect();
        let validity = Validity::from_iter((0..len).map(|index| index % 8 < 4));
        let array = PrimitiveArray::new(values, validity);
        assert_eq!(top_value(&array)?, Some((9u16.into(), len / 8 * 3)));
        Ok(())
    }

    /// Floats compare by bit pattern, so NaNs are equal and `-0.0` differs from `0.0`.
    #[test]
    fn floats_compare_bits() -> VortexResult<()> {
        let values: Buffer<f64> = [f64::NAN, -0.0, f64::NAN, 0.0, f64::NAN, -0.0, f64::NAN]
            .into_iter()
            .collect();
        let array = PrimitiveArray::new(values, Validity::NonNullable);
        let (top, count) = top_value(&array)?.vortex_expect("array has valid values");
        assert!(f64::try_from(top)?.is_nan());
        assert_eq!(count, 4);

        let values: Buffer<f64> = [0.0, -0.0, -0.0].into_iter().collect();
        let array = PrimitiveArray::new(values, Validity::NonNullable);
        let (top, count) = top_value(&array)?.vortex_expect("array has valid values");
        assert_eq!(f64::try_from(top)?.to_bits(), (-0.0f64).to_bits());
        assert_eq!(count, 2);
        Ok(())
    }

    #[rstest]
    #[case::all_valid(Validity::NonNullable)]
    #[case::nullable(Validity::from_iter((0..1_000).map(|index| index % 3 != 0)))]
    fn bytes(#[case] validity: Validity) -> VortexResult<()> {
        let values: Buffer<i8> = (0..1_000)
            .map(|index| {
                if index % 3 == 1 {
                    -5
                } else {
                    (index % 7) as i8
                }
            })
            .collect();
        // Every `-5` is valid in both cases.
        let array = PrimitiveArray::new(values, validity);
        assert_eq!(top_value(&array)?, Some(((-5i8).into(), 333)));
        Ok(())
    }

    #[test]
    fn all_null() -> VortexResult<()> {
        let array = PrimitiveArray::new(Buffer::<u32>::zeroed(10), Validity::AllInvalid);
        assert_eq!(top_value(&array)?, None);
        Ok(())
    }
}
