// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::min;
use std::ops::AddAssign;

use num_traits::AsPrimitive;
use num_traits::NumCast;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitChunkIterator;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ArrayView;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::filter::FilterKernel;
use crate::arrays::runend::RunEnd;
use crate::arrays::runend::array::RunEndArrayExt;
use crate::arrays::runend::array::RunEndArraySlotsExt;
use crate::arrays::runend::compute::take::take_indices_unchecked;
use crate::dtype::NativePType;
use crate::match_each_unsigned_integer_ptype;
use crate::validity::Validity;

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
                    ctx.allocator(),
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
    allocator: &BufferAllocatorRef,
) -> VortexResult<(PrimitiveArray, Mask)> {
    let mut filtered_run_ends = BufferMut::<R>::zeroed_in(run_ends.len(), allocator.clone());

    let mut retained_run_count = 0;
    let mut filtered_end = R::zero();
    let mut rank = MaskRank::new(mask);
    let mut selected_before_run = 0;

    let values_mask: Mask = BitBuffer::collect_bool(run_ends.len(), |run_idx| {
        let run_end = min(run_ends[run_idx].as_() - offset, length);
        let run_end_idx = run_end
            .try_into()
            .vortex_expect("run end index must fit in usize");
        // Run ends increase, so the rank walks the mask's words once over all runs instead of
        // counting each run's range from scratch, which dominates when runs are short.
        let selected_through_run = rank.rank(run_end_idx);
        let selected_in_run = selected_through_run - selected_before_run;
        selected_before_run = selected_through_run;
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

/// Counts the set bits of a mask before non-decreasing positions, reading each 64-bit word once.
struct MaskRank<'a> {
    words: BitChunkIterator<'a>,
    remainder: u64,
    /// The index of `word` among the mask's words.
    word_idx: usize,
    word: u64,
    /// The set bits in the words before `word`.
    before_word: usize,
}

impl<'a> MaskRank<'a> {
    fn new(mask: &'a BitBuffer) -> Self {
        let chunks = mask.chunks();
        let remainder = chunks.remainder_bits();
        let mut words = chunks.iter();
        let word = words.next().unwrap_or(remainder);
        Self {
            words,
            remainder,
            word_idx: 0,
            word,
            before_word: 0,
        }
    }

    /// The set bits in `0..pos`, for `pos` no smaller than in the previous call.
    #[inline]
    fn rank(&mut self, pos: usize) -> usize {
        let word_idx = pos / 64;
        while self.word_idx < word_idx {
            self.before_word += self.word.count_ones() as usize;
            self.word_idx += 1;
            self.word = self.words.next().unwrap_or(self.remainder);
        }
        let bits = pos % 64;
        let low = if bits == 0 {
            0
        } else {
            u64::MAX >> (64 - bits)
        };
        self.before_word + (self.word & low).count_ones() as usize
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::BitBuffer;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::runend::RunEnd;
    use crate::arrays::runend::RunEndArray;
    use crate::arrays::runend::compute::filter::filter_run_end_primitive;
    use crate::arrays::runend::tests::SESSION;
    use crate::assert_arrays_eq;

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

    /// The rank walk must count the same rows per run as counting each run's range, for masks
    /// that start inside a byte and end inside or at the end of a word.
    #[rstest]
    #[case(0, 128)]
    #[case(3, 128)]
    #[case(5, 200)]
    #[case(61, 64)]
    #[case(7, 1)]
    fn short_runs_count_each_runs_selected_rows(
        #[case] mask_offset: usize,
        #[case] len: usize,
    ) -> VortexResult<()> {
        let bits = BitBuffer::from_iter((0..mask_offset + len).map(|i| i % 3 != 0 || i % 7 == 0));
        let mask = bits.slice(mask_offset..mask_offset + len);
        // Runs of 1, 2 and 3 rows.
        let mut run_ends = Vec::new();
        let mut end = 0u32;
        let mut run_len = 1;
        while (end as usize) < len {
            end = (end + run_len).min(u32::try_from(len)?);
            run_ends.push(end);
            run_len = run_len % 3 + 1;
        }

        let ctx = SESSION.create_execution_ctx();
        let (filtered_ends, values_mask) =
            filter_run_end_primitive(&run_ends, 0, len as u64, &mask, ctx.allocator())?;

        let mut expected_ends = Vec::new();
        let mut expected_values = Vec::new();
        let (mut start, mut selected) = (0, 0u32);
        for &end in &run_ends {
            let in_run = mask.count_range(start, end as usize);
            selected += u32::try_from(in_run)?;
            expected_values.push(in_run > 0);
            if in_run > 0 {
                expected_ends.push(selected);
            }
            start = end as usize;
        }
        assert_eq!(filtered_ends.as_slice::<u32>(), expected_ends.as_slice());
        assert_eq!(values_mask, Mask::from_iter(expected_values));
        Ok(())
    }
}
