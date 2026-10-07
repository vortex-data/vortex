// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::min;

use num_traits::AsPrimitive;
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

/// Counts each run's selected rows directly when runs average at least this many rows.
///
/// Long runs make the per-run SIMD popcount cheap, while a prefix-count table costs a pass over
/// every mask word. A larger value favors the prefix-count table.
const COUNT_RANGE_MIN_ROWS_PER_RUN: u64 = 256;

/// Returns whether filtering `run_count` runs down to `selected_rows` rows takes rows directly
/// instead of filtering runs.
pub(crate) fn filter_uses_direct_take(selected_rows: usize, run_count: usize) -> bool {
    let selected_rows_per_run = selected_rows as f64 / run_count as f64;
    selected_rows_per_run < TAKE_SELECTED_ROWS_PER_RUN_THRESHOLD
        || selected_rows < MIN_RUN_FILTER_SELECTED_ROWS
}

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
        if filter_uses_direct_take(selected_rows, array.ends().len()) {
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
pub fn filter_run_end_primitive<R: NativePType + AsPrimitive<u64>>(
    run_ends: &[R],
    offset: u64,
    length: u64,
    mask: &BitBuffer,
) -> VortexResult<(PrimitiveArray, Mask)>
where
    usize: AsPrimitive<R>,
{
    let mut filtered_run_ends = buffer_mut![R::zero(); run_ends.len()];
    let mut retained_run_count = 0;
    let mut ranker = RunRanker::new(mask, length, run_ends.len());
    let mut selected_before = 0;

    let values_mask: Mask = BitBuffer::collect_bool(run_ends.len(), |run_idx| {
        let selected_through_run =
            ranker.selected_through(run_end_index(run_ends[run_idx], offset, length));
        let retain_run = selected_through_run > selected_before;
        selected_before = selected_through_run;

        // Always write the current end, then advance only for a retained run. This keeps the loop
        // branchless. The end is at most `length`, which the source run-end type already
        // represents, so the cast cannot truncate.
        filtered_run_ends[retained_run_count] = selected_through_run.as_();
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

/// Returns the index of a run's end within the `offset..offset + length` window.
///
/// The input contract and clamp prove the result is at most the mask length.
#[inline]
fn run_end_index<R: AsPrimitive<u64>>(run_end: R, offset: u64, length: u64) -> usize {
    min(run_end.as_() - offset, length)
        .try_into()
        .vortex_expect("run end index must fit in usize")
}

/// Returns each run's end in the filtered output, which is the number of rows `mask` selects up to
/// the end of that run.
///
/// Unlike [`filter_run_end_primitive`], runs without selected rows are kept as empty runs, so the
/// ends still line up with the source run values. The caller supplies the same validated window as
/// for [`filter_run_end_primitive`].
pub(crate) fn filtered_run_ends<R: AsPrimitive<u64>>(
    run_ends: &[R],
    offset: u64,
    length: u64,
    mask: &BitBuffer,
) -> Vec<usize> {
    let mut ranker = RunRanker::new(mask, length, run_ends.len());
    run_ends
        .iter()
        .map(|&end| ranker.selected_through(run_end_index(end, offset, length)))
        .collect()
}

/// Counts the rows a mask selects up to each run end, visiting run ends in order.
enum RunRanker<'a> {
    /// Counts each run directly; cheap when runs span many words.
    CountRange {
        mask: &'a BitBuffer,
        run_start: usize,
        selected: usize,
    },
    /// Looks up per-word prefix counts; cheap when runs span few words.
    Prefix(PrefixRank),
}

impl<'a> RunRanker<'a> {
    fn new(mask: &'a BitBuffer, length: u64, run_count: usize) -> Self {
        if length >= COUNT_RANGE_MIN_ROWS_PER_RUN * run_count as u64 {
            Self::CountRange {
                mask,
                run_start: 0,
                selected: 0,
            }
        } else {
            Self::Prefix(PrefixRank::new(mask))
        }
    }

    /// Returns the number of selected rows before `run_end`, which must not be less than the
    /// previous run end or exceed the mask length.
    #[inline]
    fn selected_through(&mut self, run_end: usize) -> usize {
        match self {
            Self::CountRange {
                mask,
                run_start,
                selected,
            } => {
                *selected += mask.count_range(*run_start, run_end);
                *run_start = run_end;
                *selected
            }
            Self::Prefix(rank) => rank.rank(run_end),
        }
    }
}

/// Counts the set bits before any position of a mask in constant time.
///
/// For short runs, counting each run with `BitBuffer::count_range` is dominated by its head, tail
/// and dispatch costs, and a streaming cursor mispredicts whenever a run crosses a word. Prefix
/// counts per word keep each lookup to two loads and a popcount.
struct PrefixRank {
    /// Mask words followed by a zero word, so a lookup at the mask length stays in bounds.
    words: Vec<u64>,
    /// Set bits before each word.
    before_word: Vec<usize>,
}

impl PrefixRank {
    fn new(mask: &BitBuffer) -> Self {
        let chunks = mask.chunks();
        let words: Vec<u64> = chunks.iter().chain([chunks.remainder_bits(), 0]).collect();
        let mut selected = 0;
        let before_word = words
            .iter()
            .map(|word| {
                let before = selected;
                selected += word.count_ones() as usize;
                before
            })
            .collect();
        Self { words, before_word }
    }

    /// Returns the number of set bits before `pos`, which must not exceed the mask length.
    #[inline]
    fn rank(&self, pos: usize) -> usize {
        let word_idx = pos / 64;
        let low_bits = self.words[word_idx] & ((1u64 << (pos % 64)) - 1);
        self.before_word[word_idx] + low_bits.count_ones() as usize
    }
}

#[cfg(test)]
mod tests {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
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

    /// Checks the run counts against a per-bit count over runs that straddle mask words, with both
    /// a bit offset into the mask buffer and a logical offset into the run ends. Runs averaging at
    /// least `COUNT_RANGE_MIN_ROWS_PER_RUN` rows count ranges directly; shorter runs use prefix
    /// counts.
    #[rstest]
    #[case::short_runs(1_000, 3, 0, 0)]
    #[case::word_multiple(1_024, 64, 0, 0)]
    #[case::mask_bit_offset(777, 9, 5, 0)]
    #[case::array_offset(500, 7, 0, 13)]
    #[case::both_offsets(4_099, 130, 63, 200)]
    #[case::single_run(70, 1_000, 1, 0)]
    #[case::long_runs(20_000, 1_000, 3, 7)]
    fn filter_run_end_matches_per_bit_count(
        #[case] length: usize,
        #[case] max_run: usize,
        #[case] mask_offset: usize,
        #[case] array_offset: usize,
    ) -> VortexResult<()> {
        let mut rng = StdRng::seed_from_u64(length as u64);

        let mut run_ends = Vec::new();
        let mut end = 0;
        while end < array_offset + length {
            end += rng.random_range(1..=max_run);
            run_ends.push(end);
        }
        // Keep only runs that end inside the window, as a sliced RunEnd does.
        run_ends.retain(|&e| e > array_offset);
        let run_ends_u64: Vec<u64> = run_ends.iter().map(|&e| e as u64).collect();

        let bits: Vec<bool> = (0..mask_offset + length)
            .map(|_| rng.random_bool(0.3))
            .collect();
        let mask = BitBuffer::from(bits.clone()).slice(mask_offset..mask_offset + length);
        let bits = &bits[mask_offset..];

        let (ends, values_mask) =
            filter_run_end_primitive(&run_ends_u64, array_offset as u64, length as u64, &mask)?;

        let mut expected_ends = Vec::new();
        let mut expected_values = Vec::new();
        let (mut start, mut selected) = (0, 0);
        for &e in &run_ends {
            let end = (e - array_offset).min(length);
            let in_run = bits[start..end].iter().filter(|&&b| b).count();
            selected += in_run;
            if in_run > 0 {
                expected_ends.push(selected as u64);
            }
            expected_values.push(in_run > 0);
            start = end;
        }

        assert_eq!(ends.as_slice::<u64>(), expected_ends.as_slice());
        assert_eq!(values_mask, Mask::from_iter(expected_values));
        Ok(())
    }
}
