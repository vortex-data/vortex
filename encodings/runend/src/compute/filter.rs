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
    let mut filtered_run_ends = buffer_mut![R::zero(); run_ends.len()];

    let mut retained_run_count = 0;
    let mut filtered_end = R::zero();
    // Runs are visited in order, so each run's popcount is the difference of the ranks of its
    // bounds, which one pass over the mask words computes. Counting each run separately costs a
    // call per run, which dominates for short runs.
    let chunks = mask.chunks();
    let mut ranks = RankCursor::new(chunks.iter_padded());
    let mut rank_at_run_start = 0usize;

    let values_mask: Mask = BitBuffer::collect_bool(run_ends.len(), |run_idx| {
        let run_end = min(run_ends[run_idx].as_() - offset, length);
        let run_end_idx = run_end
            .try_into()
            .vortex_expect("run end index must fit in usize");
        let rank_at_run_end = ranks.rank(run_end_idx);
        let selected_in_run = rank_at_run_end - rank_at_run_start;
        rank_at_run_start = rank_at_run_end;
        filtered_end += <R as NumCast>::from(selected_in_run)
            .vortex_expect("run popcount must fit in run-end native type");
        let retain_run = selected_in_run > 0;

        // Always write the current end, then advance only for a retained run. This keeps the loop
        // branchless.
        filtered_run_ends[retained_run_count] = filtered_end;
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

/// Counts the set bits before non-decreasing positions of a bitmap, given as 64-bit words.
struct RankCursor<I> {
    words: I,
    /// The word holding the bits from `word_start`.
    word: u64,
    word_start: usize,
    /// The set bits before `word_start`.
    rank_at_word_start: usize,
}

impl<I: Iterator<Item = u64>> RankCursor<I> {
    fn new(mut words: I) -> Self {
        let word = words.next().unwrap_or(0);
        Self {
            words,
            word,
            word_start: 0,
            rank_at_word_start: 0,
        }
    }

    /// The number of set bits before `position`, which must not be less than in a previous call.
    #[inline]
    fn rank(&mut self, position: usize) -> usize {
        while position >= self.word_start + 64 {
            self.rank_at_word_start += self.word.count_ones() as usize;
            self.word = self.words.next().unwrap_or(0);
            self.word_start += 64;
        }
        let bits = position - self.word_start;
        let below = if bits == 0 {
            0
        } else {
            self.word & (u64::MAX >> (64 - bits))
        };
        self.rank_at_word_start + below.count_ones() as usize
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

    /// The retained run ends and run mask that counting each run separately gives.
    fn filter_run_ends_per_run(
        run_ends: &[u32],
        offset: u64,
        length: u64,
        mask: &BitBuffer,
    ) -> (Vec<u32>, Vec<bool>) {
        let mut ends = Vec::new();
        let mut retained = Vec::new();
        let (mut start, mut end_count) = (0usize, 0u32);
        for &run_end in run_ends {
            let end = (u64::from(run_end) - offset).min(length) as usize;
            let selected = mask.count_range(start, end);
            end_count += selected as u32;
            if selected > 0 {
                ends.push(end_count);
            }
            retained.push(selected > 0);
            start = end;
        }
        (ends, retained)
    }

    #[rstest]
    fn filter_run_ends_matches_per_run_counts(
        #[values(0, 3, 64, 70)] mask_offset: usize,
        #[values(1, 3, 63, 64, 65, 200)] run_length: u32,
    ) -> VortexResult<()> {
        let length = 1000u64;
        let offset = 5;
        // Every run ends after the offset, and the last one at the end of the array.
        let last_end = (offset + length) as u32;
        let run_ends: Vec<u32> = (1..)
            .map(|run| offset as u32 + run * run_length)
            .take_while(|&end| end < last_end)
            .chain([last_end])
            .collect();
        let bits = BitBuffer::collect_bool(length as usize + mask_offset, |i| {
            (i * 7 + i / 13) % 5 < 2
        })
        .slice(mask_offset..mask_offset + length as usize);

        let (ends, values_mask) = filter_run_end_primitive(&run_ends, offset, length, &bits)?;
        let (expected_ends, expected_mask) =
            filter_run_ends_per_run(&run_ends, offset, length, &bits);
        assert_eq!(ends.as_slice::<u32>(), expected_ends.as_slice());
        assert_eq!(
            values_mask.to_bit_buffer().iter().collect::<Vec<_>>(),
            expected_mask
        );
        Ok(())
    }

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
}
