// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::min;
use std::ops::AddAssign;

use num_traits::AsPrimitive;
use num_traits::NumCast;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::filter::FilterKernel;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::buffer_mut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::RunEnd;
use crate::array::RunEndArrayExt;
use crate::array::RunEndArraySlotsExt;
use crate::compute::take::take_indices_unchecked;

/// Takes directly below this average number of selected rows per source run.
///
/// Take scales with the selection size; run filtering scans every run. [#1969] introduced this
/// heuristic without a focused threshold benchmark. A larger value favors take.
///
/// [#1969]: https://github.com/vortex-data/vortex/pull/1969
const TAKE_SELECTED_ROWS_PER_RUN_THRESHOLD: f64 = 0.1;

/// Takes directly below this row count to avoid fixed run and value filtering costs.
/// [#1969] introduced the cutoff. A larger value favors take.
///
/// [#1969]: https://github.com/vortex-data/vortex/pull/1969
const MIN_RUN_FILTER_SELECTED_ROWS: usize = 25;

impl FilterKernel for RunEnd {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let mask_values = mask
            .values()
            .vortex_expect("FilterKernel precondition: mask is Mask::Values");
        let selected_rows = mask_values.true_count();
        let source_run_count = array.ends().len();
        let selected_rows_per_run = selected_rows as f64 / source_run_count as f64;
        let use_direct_take = selected_rows_per_run < TAKE_SELECTED_ROWS_PER_RUN_THRESHOLD
            || selected_rows < MIN_RUN_FILTER_SELECTED_ROWS;

        if use_direct_take {
            return Ok(Some(take_indices_unchecked(
                array,
                mask_values.indices(),
                &Validity::NonNullable,
                ctx,
            )?));
        }

        let primitive_run_ends = array.ends().clone().execute::<PrimitiveArray>(ctx)?;
        let (filtered_run_ends, values_mask) =
            match_each_unsigned_integer_ptype!(primitive_run_ends.ptype(), |P| {
                filter_run_end_primitive(
                    primitive_run_ends.as_slice::<P>(),
                    array.offset() as u64,
                    array.len() as u64,
                    mask_values.bit_buffer(),
                )?
            });
        let filtered_values = array.values().filter(values_mask)?;

        // SAFETY: `filter_run_end_primitive` returns one strictly increasing end for each retained
        // run value, with the final end equal to `selected_rows`.
        let filtered = unsafe {
            RunEnd::new_unchecked(
                filtered_run_ends.into_array(),
                filtered_values,
                0,
                selected_rows,
            )
        };

        Ok(Some(filtered.into_array()))
    }
}

/// Recomputes cumulative run ends and selects the run values retained by `mask`.
///
/// The caller supplies validated RunEnd metadata for `offset..offset + length` and a mask containing
/// exactly `length` bits. The returned ends are strictly increasing and contain one entry for each
/// selected run value.
///
/// Adapted from the [Apache Arrow Rust implementation](https://github.com/apache/arrow-rs/blob/b1f5c250ebb6c1252b4e7c51d15b8e77f4c361fa/arrow-select/src/filter.rs#L425).
pub fn filter_run_end_primitive<R: NativePType + AddAssign + From<bool> + AsPrimitive<u64>>(
    run_ends: &[R],
    offset: u64,
    length: u64,
    mask: &BitBuffer,
) -> VortexResult<(PrimitiveArray, Mask)> {
    if mask.len() >= run_ends.len().saturating_mul(BULK_COUNT_MIN_RUN_LENGTH) {
        let mut position = 0;
        let mut counted = 0;
        return filter_runs(run_ends, offset, length, |end| {
            counted += mask.count_range(position, end);
            position = end;
            counted
        });
    }
    let mut ranks = WordRanks::new(mask);
    filter_runs(run_ends, offset, length, |end| ranks.rank(end))
}

/// Filters run ends given `rank`, the set bits of the mask before each increasing run end.
fn filter_runs<R: NativePType + AsPrimitive<u64>>(
    run_ends: &[R],
    offset: u64,
    length: u64,
    mut rank: impl FnMut(usize) -> usize,
) -> VortexResult<(PrimitiveArray, Mask)> {
    let mut filtered_run_ends = buffer_mut![R::zero(); run_ends.len()];
    let mut run_start_rank = 0;
    let mut retained_run_count = 0;

    let values_mask: Mask = BitBuffer::collect_bool(run_ends.len(), |run_idx| {
        let run_end = min(run_ends[run_idx].as_() - offset, length);
        let run_end_rank = rank(
            run_end
                .try_into()
                .vortex_expect("run end index must fit in usize"),
        );
        let retain_run = run_end_rank > run_start_rank;
        run_start_rank = run_end_rank;

        // Always write the current end, then advance only for a retained run. This keeps the loop
        // branchless.
        filtered_run_ends[retained_run_count] = <R as NumCast>::from(run_end_rank)
            .vortex_expect("run popcount must fit in run-end native type");
        retained_run_count += retain_run as usize;

        retain_run
    })
    .into();

    filtered_run_ends.truncate(retained_run_count);

    Ok((
        PrimitiveArray::new(filtered_run_ends, Validity::NonNullable),
        values_mask,
    ))
}

/// Average run length from which one bulk SIMD popcount per run beats [`WordRanks`].
const BULK_COUNT_MIN_RUN_LENGTH: usize = 256;

/// Counts the set bits before increasing positions of a mask in one pass over its words.
///
/// A run's selected rows are the difference of the ranks at its ends, so each short run costs a
/// word popcount rather than a bulk popcount call.
struct WordRanks {
    words: Vec<u64>,
    /// The words whose bits `counted` covers.
    word: usize,
    /// Set bits in `words[..word]`.
    counted: usize,
}

impl WordRanks {
    fn new(mask: &BitBuffer) -> Self {
        let chunks = mask.chunks();
        let mut words: Vec<u64> = chunks.iter().collect();
        if chunks.remainder_len() > 0 {
            words.push(chunks.remainder_bits());
        }
        Self {
            words,
            word: 0,
            counted: 0,
        }
    }

    /// The set bits before `position`. Positions must not decrease between calls.
    fn rank(&mut self, position: usize) -> usize {
        let word = position / 64;
        while self.word < word {
            self.counted += self.words[self.word].count_ones() as usize;
            self.word += 1;
        }
        let bits = position % 64;
        if bits == 0 {
            return self.counted;
        }
        self.counted + (self.words[word] & ((1u64 << bits) - 1)).count_ones() as usize
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_buffer::BitBuffer;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use super::filter_run_end_primitive;
    use crate::RunEnd;
    use crate::RunEndArray;
    use crate::tests::SESSION;

    fn ree_array() -> RunEndArray {
        RunEnd::encode(
            PrimitiveArray::from_iter([1, 1, 1, 4, 4, 4, 2, 2, 5, 5, 5, 5]).into_array(),
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap()
    }

    #[test]
    fn filter_sliced_run_end() -> VortexResult<()> {
        let arr = ree_array().slice(2..7)?;
        let filtered = arr.filter(Mask::from_iter([true, false, false, true, true]))?;

        let mut ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(filtered, PrimitiveArray::from_iter([1i32, 4, 2]), &mut ctx);
        Ok(())
    }

    /// Regression: Filter(Slice(RunEnd)) must preserve RunEnd after execution.
    /// Previously Filter.execute() forced its child to canonical, decoding
    /// Slice(RunEnd) → Primitive and destroying run structure. The fix lets
    /// Filter unwrap one layer at a time so RunEnd's FilterKernel can fire.
    #[test]
    fn filter_sliced_run_end_preserves_encoding() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();

        // 4 runs of 32 each = 128 rows. Large enough that FilterKernel takes
        // the run-preserving path (true_count >= 25).
        let values: Vec<i32> = [10, 20, 30, 40]
            .iter()
            .flat_map(|&v| std::iter::repeat_n(v, 32))
            .collect();
        let arr = RunEnd::encode(PrimitiveArray::from_iter(values).into_array(), &mut ctx)?;

        // Slice off the first 16 rows. Slice(RunEnd), 112 rows, 4 runs.
        let sliced = arr.into_array().slice(16..128)?;

        // Keep every other row = 112/2 = 56 rows.
        let mask = Mask::from_iter((0..sliced.len()).map(|i| i % 2 == 0));
        let filtered = sliced.filter(mask)?;

        let executed = filtered.execute_until::<RunEnd>(&mut ctx)?;
        assert_eq!(
            executed.encoding_id().as_ref(),
            "vortex.runend",
            "Filter(Slice(RunEnd)) should preserve RunEnd encoding"
        );

        let expected: Vec<i32> = std::iter::repeat_n(10, 8)
            .chain(std::iter::repeat_n(20, 16))
            .chain(std::iter::repeat_n(30, 16))
            .chain(std::iter::repeat_n(40, 16))
            .collect();
        assert_arrays_eq!(executed, PrimitiveArray::from_iter(expected), &mut ctx);

        Ok(())
    }

    /// Both ways of counting a run's selected rows agree with a bit-by-bit count, including masks
    /// that start inside a byte, end inside a word, and run ends past the array's length.
    #[rstest]
    #[case::short_runs(3)]
    #[case::word_runs(64)]
    #[case::bulk_runs(300)]
    fn filter_run_end_counts_runs(#[case] run_length: u32) -> VortexResult<()> {
        let (offset, length, mask_offset) = (5u32, 1_000u32, 3usize);
        let run_ends: Vec<u32> = (1..)
            .map(|run| offset + run * run_length - run % 2)
            .take_while(|&end| end < offset + length + run_length)
            .collect();
        let bits: Vec<bool> = (0..mask_offset + length as usize)
            .map(|i| i % 7 == 0 || i % 5 == 1)
            .collect();
        let mask = BitBuffer::from(bits.clone()).slice(mask_offset..mask_offset + length as usize);

        let (ends, values_mask) =
            filter_run_end_primitive(&run_ends, offset.into(), length.into(), &mask)?;

        let mut expected_ends = Vec::new();
        let mut expected_values = Vec::new();
        let (mut start, mut selected) = (0usize, 0u32);
        for &end in &run_ends {
            let end = ((end - offset) as usize).min(length as usize);
            let in_run = u32::try_from(
                bits[mask_offset + start..mask_offset + end]
                    .iter()
                    .filter(|&&bit| bit)
                    .count(),
            )?;
            selected += in_run;
            if in_run > 0 {
                expected_ends.push(selected);
            }
            expected_values.push(in_run > 0);
            start = end;
        }
        assert_eq!(ends.as_slice::<u32>(), expected_ends.as_slice());
        assert_eq!(values_mask, Mask::from_iter(expected_values));
        Ok(())
    }
}
