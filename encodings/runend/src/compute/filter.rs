// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::min;

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
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
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

/// Filters through the sparse set-bit walk below this many selected rows per source run.
///
/// The walk costs one step per selected row on top of the run scan, while the rank scan costs a
/// fixed lookup per run. The `run_end_filter` benchmark crosses over near 0.3 selected rows per run.
const SPARSE_SELECTED_ROWS_PER_RUN_THRESHOLD: f64 = 0.25;

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
                let run_ends = primitive_run_ends.as_slice::<P>();
                let offset = array.offset() as u64;
                if selected_rows_per_run < SPARSE_SELECTED_ROWS_PER_RUN_THRESHOLD {
                    filter_run_end_sparse(run_ends, offset, mask_values.bit_buffer())?
                } else {
                    filter_run_end_primitive(
                        run_ends,
                        offset,
                        array.len() as u64,
                        mask_values.bit_buffer(),
                    )?
                }
            });
        let filtered_values = array.values().filter(values_mask)?;

        // SAFETY: both run filters return one strictly increasing end for each retained
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
/// Each run's retained length is a difference of mask ranks at its bounds; per-word cumulative
/// popcounts make each rank a lookup plus one masked popcount.
pub fn filter_run_end_primitive<R: NativePType + AsPrimitive<u64>>(
    run_ends: &[R],
    offset: u64,
    length: u64,
    mask: &BitBuffer,
) -> VortexResult<(PrimitiveArray, Mask)> {
    // The trailing zero word lets `rank(length)` index one word past a multiple-of-64 length.
    let words: Vec<u64> = mask
        .chunks()
        .iter_padded()
        .chain(std::iter::once(0))
        .collect();
    let mut word_ranks = Vec::with_capacity(words.len());
    let mut rank = 0usize;
    for word in &words {
        word_ranks.push(rank);
        rank += word.count_ones() as usize;
    }

    let mut filtered_run_ends = BufferMut::<R>::with_capacity(run_ends.len());
    let mut previous_rank = 0usize;
    let values_mask: Mask = BitBuffer::collect_bool(run_ends.len(), |run_idx| {
        let run_end: usize = min(run_ends[run_idx].as_() - offset, length).as_();
        let word = run_end / 64;
        let rank =
            word_ranks[word] + (words[word] & ((1u64 << (run_end % 64)) - 1)).count_ones() as usize;
        let retain_run = rank > previous_rank;
        previous_rank = rank;

        // Always write the current end, then keep it only for a retained run. This keeps the loop
        // branchless.
        // SAFETY: the capacity is `run_ends.len()` and the length grows by at most one per run.
        unsafe {
            filtered_run_ends.push_unchecked(
                <R as NumCast>::from(rank).vortex_expect("filtered end must fit in run-end type"),
            );
            let dropped = <usize as From<bool>>::from(!retain_run);
            filtered_run_ends.set_len(filtered_run_ends.len() - dropped);
        }
        retain_run
    })
    .into();

    Ok((
        PrimitiveArray::new(filtered_run_ends, Validity::NonNullable),
        values_mask,
    ))
}

/// Recomputes run ends for a sparse `mask` by walking its set bits instead of every run's rank.
///
/// Each selected row advances a cursor over `run_ends`, so the cost is a comparison per run plus
/// one step per selected row. Same contract and output as [`filter_run_end_primitive`].
pub fn filter_run_end_sparse<R: NativePType + AsPrimitive<u64>>(
    run_ends: &[R],
    offset: u64,
    mask: &BitBuffer,
) -> VortexResult<(PrimitiveArray, Mask)> {
    let mut filtered_run_ends =
        BufferMut::<R>::with_capacity(mask.true_count().min(run_ends.len()));
    let mut values_mask = BitBufferMut::new_unset(run_ends.len());

    let mut run_idx = 0usize;
    let mut current_run = usize::MAX;
    let mut selected = 0usize;
    mask.for_each_set_index(|idx| {
        let position = idx as u64 + offset;
        while run_ends[run_idx].as_() <= position {
            run_idx += 1;
        }
        if run_idx != current_run {
            if current_run != usize::MAX {
                filtered_run_ends.push(
                    <R as NumCast>::from(selected)
                        .vortex_expect("filtered end must fit in run-end type"),
                );
            }
            values_mask.set(run_idx);
            current_run = run_idx;
        }
        selected += 1;
    });
    if selected > 0 {
        filtered_run_ends.push(
            <R as NumCast>::from(selected).vortex_expect("filtered end must fit in run-end type"),
        );
    }

    Ok((
        PrimitiveArray::new(filtered_run_ends, Validity::NonNullable),
        Mask::from(values_mask.freeze()),
    ))
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
    use super::filter_run_end_sparse;
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
        assert_arrays_eq!(
            filtered,
            RunEnd::new(
                PrimitiveArray::from_iter([1u8, 2, 3]).into_array(),
                PrimitiveArray::from_iter([1i32, 4, 2]).into_array(),
                &mut ctx,
            ),
            &mut ctx
        );
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

    /// Both run filters must match a per-row reference, including sliced offsets, runs that
    /// straddle the slice bounds, and lengths on and off 64-bit word boundaries.
    #[rstest]
    #[case(0, 128)]
    #[case(0, 130)]
    #[case(5, 64)]
    #[case(37, 200)]
    #[case(100, 1)]
    fn run_filters_match_reference(
        #[case] offset: usize,
        #[case] length: usize,
    ) -> VortexResult<()> {
        // Runs of length 1..=7 covering `0..offset + length + 10`.
        let total = offset + length + 10;
        let mut all_run_ends = Vec::new();
        let mut end = 0u32;
        let mut step = 0u32;
        while (end as usize) < total {
            step = step % 7 + 1;
            end += step;
            all_run_ends.push(end);
        }
        // Keep only the runs overlapping `offset..offset + length`, as a RunEnd slice does.
        let first = all_run_ends.partition_point(|&e| e as usize <= offset);
        let last = all_run_ends.partition_point(|&e| (e as usize) < offset + length);
        let run_ends = &all_run_ends[first..=last];
        let run_of = |row: usize| run_ends.iter().position(|&e| e as usize > row + offset);

        for modulus in [1usize, 2, 3, 11, 64] {
            let mask = BitBuffer::from_iter((0..length).map(|i| (i * 7 + 3) % modulus == 0));

            let mut expected_ends = Vec::new();
            let mut expected_runs = Vec::new();
            let mut selected = 0u32;
            for row in (0..length).filter(|&i| mask.value(i)) {
                let run = run_of(row);
                if expected_runs.last() != Some(&run) {
                    if !expected_runs.is_empty() {
                        expected_ends.push(selected);
                    }
                    expected_runs.push(run);
                }
                selected += 1;
            }
            if selected > 0 {
                expected_ends.push(selected);
            }
            let expected_mask =
                Mask::from_iter((0..run_ends.len()).map(|r| expected_runs.contains(&Some(r))));

            for (ends, values_mask) in [
                filter_run_end_primitive(run_ends, offset as u64, length as u64, &mask)?,
                filter_run_end_sparse(run_ends, offset as u64, &mask)?,
            ] {
                assert_eq!(ends.as_slice::<u32>(), expected_ends.as_slice());
                assert_eq!(values_mask, expected_mask);
            }
        }
        Ok(())
    }
}
