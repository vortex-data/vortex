// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Filtering run-end encoded arrays.
//!
//! Small or sparse selections use direct take. Otherwise, filtering updates the run ends and
//! removes runs that contain no selected rows, preserving the run-end encoding.
//!
//! A run's new end is the number of selected rows before its original end, called the mask's
//! _rank_. A run disappears when this count does not increase:
//!
//! ```text
//! Input rows:       A A A | B B | C C C
//! Input ends:           3     5       8
//! Selection mask:   1 0 1 | 0 0 | 1 0 1
//! Rank at each end:     2     2       4
//!
//! Output ends:      [2, 4]
//! Output values:    [A, C]
//! Output rows:      A A C C
//! ```
//!
//! Run ends never decrease, so one walk over the mask's 64-bit words answers every rank query.
//! The walk keeps the number of selected rows in the words it has passed, and a query adds a
//! population count of the relevant bits within the current word. This avoids a separate range
//! count for every run, which is especially useful for short runs, and it builds no index, so
//! long runs cost no more than one pass over the mask.
//!
//! Each query still needs a population count, so the instructions used for counting matter.
//! [`fearless_simd`] selects a version of the loop compiled for the current CPU's features.
//! On supported x86 CPUs, this enables POPCNT, which counts the set bits in a word with one
//! instruction. The rank queries themselves operate on scalar words rather than vectors.

use std::cmp::min;

use fearless_simd::Level;
use fearless_simd::Simd;
use fearless_simd::dispatch;
use fearless_simd_macros::simd;
use num_traits::AsPrimitive;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitChunkIterator;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
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
use crate::memory::BufferAllocatorRef;
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
/// A filtered run end is the rank of the mask at the source run end, i.e. the number of selected
/// rows before it, and a run is retained when its rank exceeds the previous run's rank.
pub fn filter_run_end_primitive<R: NativePType + AsPrimitive<u64>>(
    run_ends: &[R],
    offset: u64,
    length: u64,
    mask: &BitBuffer,
    allocator: &BufferAllocatorRef,
) -> VortexResult<(PrimitiveArray, Mask)>
where
    u64: AsPrimitive<R>,
{
    // The rank loop reads the mask without bounds checks and relies on this length.
    vortex_ensure_eq!(mask.len() as u64, length);

    let (filtered_run_ends, retained_runs) =
        run_end_ranks(run_ends, offset, length, mask, allocator);

    Ok((
        PrimitiveArray::new(filtered_run_ends, Validity::NonNullable),
        Mask::from(retained_runs),
    ))
}

/// Returns the rank of `mask` at every retained source run end and the runs that contain selected
/// rows. `mask` must contain exactly `length` bits.
fn run_end_ranks<R: NativePType + AsPrimitive<u64>>(
    run_ends: &[R],
    offset: u64,
    length: u64,
    mask: &BitBuffer,
    allocator: &BufferAllocatorRef,
) -> (BufferMut<R>, BitBuffer)
where
    u64: AsPrimitive<R>,
{
    // The x86-64 baseline has no POPCNT, so a plain build lowers every per-run popcount to a
    // bit-twiddling sequence that dominates the loop. Levels from SSE4.2 up enable POPCNT.
    dispatch!(Level::new(), simd => {
        run_end_ranks_simd(simd, run_ends, offset, length, mask, allocator)
    })
}

/// Rank loop compiled with the target features of `simd`.
///
/// Each level has its own instantiation, so each `BitBuffer::collect_bool` instantiation has one
/// caller and LLVM inlines it into the feature-enabled body. An out-of-line `collect_bool` would
/// compile the predicate without POPCNT.
///
/// The loop writes every rank at the current retained count and advances the count only for a
/// retained run, so it compacts the ends without a branch or a second pass.
#[simd]
fn run_end_ranks_simd<S: Simd, R: NativePType + AsPrimitive<u64>>(
    _: S,
    run_ends: &[R],
    offset: u64,
    length: u64,
    mask: &BitBuffer,
    allocator: &BufferAllocatorRef,
) -> (BufferMut<R>, BitBuffer)
where
    u64: AsPrimitive<R>,
{
    let mut rank = MaskRank::new(mask);
    let mut filtered_run_ends = BufferMut::<R>::with_capacity_in(run_ends.len(), allocator.clone());
    let filtered_run_ends_out = filtered_run_ends.spare_capacity_mut();
    let mut retained_run_count = 0;
    let mut prev_rank = 0u64;

    let retained_runs = BitBuffer::collect_bool(run_ends.len(), |run_idx| {
        // SAFETY: `collect_bool` calls this closure with `run_idx < run_ends.len()`.
        let run_end = unsafe { *run_ends.get_unchecked(run_idx) };

        let run_end: u64 = run_end.as_();
        let position = <u64 as AsPrimitive<usize>>::as_(min(run_end - offset, length));

        // The clamp keeps the position at most `length`, which is the mask length, and run ends
        // never decrease, so the positions never decrease.
        let run_rank: u64 = rank.rank(position).as_();

        // SAFETY: `retained_run_count <= run_idx`, and the spare capacity holds `run_ends.len()`
        // elements. The rank is at most `length`, which fits in `R` because the source run ends
        // reach `offset + length`.
        unsafe { filtered_run_ends_out.get_unchecked_mut(retained_run_count) }
            .write(run_rank.as_());

        let retain_run = run_rank > prev_rank;
        retained_run_count += usize::from(retain_run);
        prev_rank = run_rank;
        retain_run
    });

    // SAFETY: the loop initialized the first `retained_run_count` elements.
    unsafe { filtered_run_ends.set_len(retained_run_count) };

    (filtered_run_ends, retained_runs)
}

/// Counts the set bits of a mask before non-decreasing positions, reading each 64-bit word once.
///
/// The methods are always inlined, so the rank loop counts with the POPCNT of its dispatch level.
struct MaskRank<'a> {
    words: BitChunkIterator<'a>,

    /// The last, partial word of the mask. It is zero when the mask length is a multiple of 64.
    remainder: u64,

    /// The index of `word` among the mask's words.
    word_idx: usize,

    word: u64,

    /// The set bits in the words before `word`.
    before_word: usize,
}

impl<'a> MaskRank<'a> {
    #[expect(
        clippy::inline_always,
        reason = "must inline into the dispatch body to use its POPCNT"
    )]
    #[inline(always)]
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

    /// Returns the set bits in `0..pos`, for `pos` at most the mask length and no smaller than in
    /// the previous call.
    #[expect(
        clippy::inline_always,
        reason = "must inline into the dispatch body to use its POPCNT"
    )]
    #[inline(always)]
    fn rank(&mut self, pos: usize) -> usize {
        let word_idx = pos / 64;
        while self.word_idx < word_idx {
            self.before_word += self.word.count_ones() as usize;
            self.word_idx += 1;
            self.word = self.words.next().unwrap_or(self.remainder);
        }

        let low_bits = (1u64 << (pos % 64)) - 1;
        self.before_word + (self.word & low_bits).count_ones() as usize
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::BitBuffer;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use super::filter_run_end_primitive;
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

    /// The rank walk must count the same rows per run as counting each run's range. The cases
    /// cover masks that start inside a byte and end inside or at the end of a word, and sliced
    /// arrays whose first run starts before the slice and whose last run ends after it.
    #[rstest]
    #[case(0, 0, 128)]
    #[case(0, 3, 128)]
    #[case(0, 5, 200)]
    #[case(0, 61, 64)]
    #[case(0, 7, 1)]
    #[case(3, 5, 200)]
    #[case(70, 0, 129)]
    fn filter_run_end_primitive_counts_each_runs_selected_rows(
        #[case] offset: u64,
        #[case] mask_offset: usize,
        #[case] len: usize,
    ) -> VortexResult<()> {
        let bits = BitBuffer::from_iter((0..mask_offset + len).map(|i| i % 3 != 0 || i % 7 == 0));
        let mask = bits.slice(mask_offset..mask_offset + len);

        // Runs of 1, 2, 3 and 150 rows, so the walk also skips several words in one query.
        let slice_end = offset + len as u64;
        let mut run_ends = Vec::new();
        let mut end = 0u64;
        for run_len in [1, 2, 3, 150].into_iter().cycle() {
            end += run_len;
            if end > offset {
                run_ends.push(u32::try_from(end)?);
            }
            if end >= slice_end {
                break;
            }
        }

        // End the last run after the slice, as in a slice of a longer array.
        if let Some(last) = run_ends.last_mut() {
            *last += 5;
        }

        let (filtered_ends, values_mask) = filter_run_end_primitive(
            &run_ends,
            offset,
            len as u64,
            &mask,
            SESSION.create_execution_ctx().allocator(),
        )?;

        let mut expected_ends = Vec::new();
        let mut expected_values = Vec::new();
        let (mut start, mut selected) = (0, 0u32);
        for &end in &run_ends {
            let end = usize::try_from(u64::from(end) - offset)?.min(len);
            let in_run = mask.count_range(start, end);
            selected += u32::try_from(in_run)?;
            expected_values.push(in_run > 0);
            if in_run > 0 {
                expected_ends.push(selected);
            }
            start = end;
        }

        assert_eq!(filtered_ends.as_slice::<u32>(), expected_ends.as_slice());
        assert_eq!(values_mask, Mask::from_iter(expected_values));
        Ok(())
    }
}
