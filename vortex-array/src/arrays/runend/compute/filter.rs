// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::min;
use std::ops::AddAssign;

use num_traits::AsPrimitive;
use num_traits::NumCast;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
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

/// The average number of rows per run above which the filter counts each run's bits directly.
const LONG_RUN_ROWS: u64 = 64;

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
                    selected_rows,
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
    selected_rows: usize,
    allocator: &BufferAllocatorRef,
) -> VortexResult<(PrimitiveArray, Mask)> {
    // A mask that keeps fewer rows than there are runs, such as the mask of a selective dynamic
    // filter, is cheaper to walk by its set bits than run by run.
    if selected_rows < run_ends.len() {
        return Ok(filter_run_end_sparse(
            run_ends,
            offset,
            length,
            mask,
            selected_rows,
            allocator,
        ));
    }

    // Short runs read their counts from a rank table built once. Long runs count their own bits,
    // which bulk popcount does quickly, rather than pay for the table.
    if length / LONG_RUN_ROWS < run_ends.len() as u64 {
        let rank = MaskRank::new(mask);
        let mut selected_before_run = 0;
        Ok(filter_runs(
            run_ends,
            offset,
            length,
            allocator,
            |_, run_end| {
                let selected_before_end = rank.rank(run_end);
                let selected_in_run = selected_before_end - selected_before_run;
                selected_before_run = selected_before_end;
                selected_in_run
            },
        ))
    } else {
        Ok(filter_runs(
            run_ends,
            offset,
            length,
            allocator,
            |run_start, run_end| mask.count_range(run_start, run_end),
        ))
    }
}

/// Recomputes the run ends of the runs that keep any row, given the number of rows each run keeps
/// from `selected_in_run(run_start, run_end)`.
fn filter_runs<R: NativePType + AddAssign + From<bool> + AsPrimitive<u64>>(
    run_ends: &[R],
    offset: u64,
    length: u64,
    allocator: &BufferAllocatorRef,
    mut selected_in_run: impl FnMut(usize, usize) -> usize,
) -> (PrimitiveArray, Mask) {
    let mut filtered_run_ends = BufferMut::<R>::zeroed_in(run_ends.len(), allocator.clone());

    let mut run_start_idx = 0usize;
    let mut retained_run_count = 0;
    let mut filtered_end = R::zero();

    let values_mask: Mask = BitBuffer::collect_bool(run_ends.len(), |run_idx| {
        let run_end = min(run_ends[run_idx].as_() - offset, length);
        // The input contract and clamp prove `run_start_idx <= run_end_idx <= mask.len()`.
        let run_end_idx = run_end
            .try_into()
            .vortex_expect("run end index must fit in usize");
        let selected = selected_in_run(run_start_idx, run_end_idx);
        run_start_idx = run_end_idx;
        filtered_end += <R as NumCast>::from(selected)
            .vortex_expect("run popcount must fit in run-end native type");
        let retain_run = selected > 0;

        // Always write the current end, then advance only for a retained run. This keeps the loop
        // branchless.
        filtered_run_ends[retained_run_count] = filtered_end;
        retained_run_count += retain_run as usize;

        retain_run
    })
    .into();

    filtered_run_ends.truncate(retained_run_count);

    (
        PrimitiveArray::new(filtered_run_ends, Validity::NonNullable),
        values_mask,
    )
}

/// [`filter_run_end_primitive`] for a mask with fewer set bits than runs. Each set bit finds its
/// run by galloping forward from the run of the previous set bit.
fn filter_run_end_sparse<R: NativePType + AddAssign + From<bool> + AsPrimitive<u64>>(
    run_ends: &[R],
    offset: u64,
    length: u64,
    mask: &BitBuffer,
    selected_rows: usize,
    allocator: &BufferAllocatorRef,
) -> (PrimitiveArray, Mask) {
    let run_end = |run_idx: usize| min(run_ends[run_idx].as_() - offset, length);

    let mut filtered_run_ends = BufferMut::<R>::with_capacity_in(selected_rows, allocator.clone());
    let mut values_mask = BitBufferMut::new_unset(run_ends.len());
    let mut run_idx = 0;
    let mut current_run = None;
    let mut selected = R::zero();
    for position in mask.set_indices() {
        let position = position as u64;
        if run_end(run_idx) <= position {
            // Gallop to a run that ends after the position, then search the last step.
            let mut low = run_idx;
            let mut step = 1;
            while low + step < run_ends.len() && run_end(low + step) <= position {
                low += step;
                step *= 2;
            }
            let high = (low + step).min(run_ends.len());
            run_idx = low
                + 1
                + run_ends[low + 1..high]
                    .partition_point(|end| min(end.as_() - offset, length) <= position);
        }
        if current_run != Some(run_idx) {
            if current_run.is_some() {
                filtered_run_ends.push(selected);
            }
            values_mask.set(run_idx);
            current_run = Some(run_idx);
        }
        selected += <R as From<bool>>::from(true);
    }
    if current_run.is_some() {
        filtered_run_ends.push(selected);
    }

    (
        PrimitiveArray::new(filtered_run_ends, Validity::NonNullable),
        values_mask.freeze().into(),
    )
}

/// The number of set bits of a mask before each position.
///
/// Runs can be a few rows long, so counting each run's bits separately pays the setup of a count
/// for every run. Counting every word of the mask once and keeping the counts before each word
/// makes the count before any position one table lookup and one popcount.
struct MaskRank {
    words: Vec<u64>,
    /// The number of set bits in the words before each word.
    before_word: Vec<usize>,
}

impl MaskRank {
    fn new(mask: &BitBuffer) -> Self {
        let chunks = mask.chunks();
        let mut words = chunks.iter().collect::<Vec<_>>();
        if chunks.remainder_len() > 0 {
            words.push(chunks.remainder_bits());
        }
        let before_word = words
            .iter()
            .scan(0usize, |count, word| {
                let before = *count;
                *count += word.count_ones() as usize;
                Some(before)
            })
            .collect();
        Self { words, before_word }
    }

    /// The number of set bits before `position`, which is at most the length of the mask.
    #[inline]
    fn rank(&self, position: usize) -> usize {
        let word = position / 64;
        let bit = position % 64;
        if bit == 0 {
            return match self.before_word.get(word) {
                Some(before) => *before,
                None => self.before_word.last().map_or(0, |before| {
                    before + self.words[word - 1].count_ones() as usize
                }),
            };
        }
        self.before_word[word] + (self.words[word] & ((1u64 << bit) - 1)).count_ones() as usize
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::BitBuffer;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use super::MaskRank;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::runend::RunEnd;
    use crate::arrays::runend::RunEndArray;
    use crate::arrays::runend::tests::SESSION;
    use crate::assert_arrays_eq;

    fn ree_array() -> RunEndArray {
        RunEnd::encode(
            PrimitiveArray::from_iter([1, 1, 1, 4, 4, 4, 2, 2, 5, 5, 5, 5]).into_array(),
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap()
    }

    /// The rank of every position agrees with counting the bits before it, for masks whose bits
    /// start inside a byte and whose length is not a multiple of a word.
    #[rstest]
    #[case::empty(0, 0)]
    #[case::one_word(0, 64)]
    #[case::partial_word(3, 70)]
    #[case::several_words(5, 300)]
    fn mask_rank_counts_bits_before_each_position(#[case] offset: usize, #[case] len: usize) {
        let bits = BitBuffer::from_iter((0..offset + len).map(|i| (i * 7919) % 5 < 2));
        let mask = bits.slice(offset..offset + len);
        let rank = MaskRank::new(&mask);
        for position in 0..=len {
            assert_eq!(
                rank.rank(position),
                mask.count_range(0, position),
                "{position}"
            );
        }
    }

    /// Sparse and dense masks select the same rows as filtering the decoded values.
    #[rstest]
    #[case::sparse(1)]
    #[case::dense(3)]
    fn filter_matches_decoded_values(#[case] keep_every: usize) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = (0..1000i32).map(|v| v / 7).collect::<Vec<_>>();
        let arr = RunEnd::encode(
            PrimitiveArray::from_iter(values.clone()).into_array(),
            &mut ctx,
        )?
        .into_array()
        .slice(5..995)?;
        // A sparse mask keeps fewer rows than there are runs.
        let keep = |i: usize| (i * 31) % (7 * 13) < keep_every * 7;
        let filtered = arr.filter(Mask::from_iter((0..990).map(keep)))?;
        let expected = values[5..995]
            .iter()
            .enumerate()
            .filter(|(i, _)| keep(*i))
            .map(|(_, v)| *v)
            .collect::<Vec<_>>();
        assert_arrays_eq!(filtered, PrimitiveArray::from_iter(expected), &mut ctx);
        Ok(())
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
