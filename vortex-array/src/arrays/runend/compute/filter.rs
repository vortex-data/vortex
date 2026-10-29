// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::min;

use num_traits::AsPrimitive;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
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
                    ctx,
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
/// rows before it, and a run is retained when its rank exceeds the previous run's rank. Ranks are
/// written for every run so the loop carries no data-dependent store index, and the retained ends
/// are then compacted by the primitive filter with the same mask that selects the run values.
pub fn filter_run_end_primitive<R: NativePType + AsPrimitive<u64>>(
    run_ends: &[R],
    offset: u64,
    length: u64,
    mask: &BitBuffer,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(PrimitiveArray, Mask)>
where
    u64: AsPrimitive<R>,
{
    let (ranks, retained_runs) = run_end_ranks(run_ends, offset, length, mask);
    let values_mask = Mask::from(retained_runs);

    let filtered_run_ends = PrimitiveArray::new(ranks, Validity::NonNullable)
        .into_array()
        .filter(values_mask.clone())?
        .execute::<PrimitiveArray>(ctx)?;

    Ok((filtered_run_ends, values_mask))
}

/// Returns the rank of `mask` at every source run end and the runs that contain selected rows.
fn run_end_ranks<R: NativePType + AsPrimitive<u64>>(
    run_ends: &[R],
    offset: u64,
    length: u64,
    mask: &BitBuffer,
) -> (Buffer<R>, BitBuffer)
where
    u64: AsPrimitive<R>,
{
    // The x86-64 baseline has no POPCNT, so a plain build lowers every per-run popcount to a
    // bit-twiddling sequence that dominates the loop.
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("popcnt") {
            // SAFETY: POPCNT support was detected above.
            return unsafe { run_end_ranks_popcnt(run_ends, offset, length, mask) };
        }
    }
    run_end_ranks_inline::<R, false>(run_ends, offset, length, mask)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "popcnt")]
unsafe fn run_end_ranks_popcnt<R: NativePType + AsPrimitive<u64>>(
    run_ends: &[R],
    offset: u64,
    length: u64,
    mask: &BitBuffer,
) -> (Buffer<R>, BitBuffer)
where
    u64: AsPrimitive<R>,
{
    run_end_ranks_inline::<R, true>(run_ends, offset, length, mask)
}

/// Rank loop shared by the POPCNT and portable paths.
///
/// `POPCNT` is unused in the body. It gives each path its own instantiation, and so its own
/// closure type, so each `BitBuffer::collect_bool` instantiation has one caller and LLVM inlines it.
/// With a shared closure type LLVM keeps `collect_bool` out of line, where the predicate is
/// compiled without POPCNT.
#[expect(
    clippy::inline_always,
    reason = "the loop must inline into the POPCNT-enabled caller to use the instruction"
)]
#[inline(always)]
fn run_end_ranks_inline<R: NativePType + AsPrimitive<u64>, const POPCNT: bool>(
    run_ends: &[R],
    offset: u64,
    length: u64,
    mask: &BitBuffer,
) -> (Buffer<R>, BitBuffer)
where
    u64: AsPrimitive<R>,
{
    let rank = RankDirectory::new(mask);
    let mut ranks = BufferMut::<R>::zeroed(run_ends.len());
    let mut prev_rank = 0u64;

    let retained_runs = BitBuffer::collect_bool(run_ends.len(), |run_idx| {
        let run_rank = rank.rank(min(run_ends[run_idx].as_() - offset, length));
        // The rank is at most `length`, which fits in `R` because the source run ends reach
        // `offset + length`.
        ranks[run_idx] = run_rank.as_();
        let retain_run = run_rank > prev_rank;
        prev_rank = run_rank;
        retain_run
    });

    (ranks.freeze(), retained_runs)
}

/// Constant-time rank queries over a bitmap.
///
/// Each entry pairs a 64-bit word of the bitmap with the number of set bits in all preceding
/// words, so a rank is one load and one popcount instead of a range count per query.
struct RankDirectory {
    /// `(word, set bits before word)` over the bitmap's aligned words, followed by one zero word so
    /// `rank(len)` is in bounds.
    entries: Vec<(u64, u64)>,
    /// Bit position of logical bit 0 within the first word. The padding bits are zero.
    lead_padding: u64,
}

impl RankDirectory {
    #[expect(
        clippy::inline_always,
        reason = "must inline into the POPCNT-enabled caller to use the instruction"
    )]
    #[inline(always)]
    fn new(bits: &BitBuffer) -> Self {
        let chunks = bits.unaligned_chunks();
        let mut entries = Vec::with_capacity(chunks.chunks().len() + 3);
        let mut preceding = 0u64;
        let mut push = |word: u64| {
            entries.push((word, preceding));
            preceding += u64::from(word.count_ones());
        };
        chunks.prefix().into_iter().for_each(&mut push);
        chunks.chunks().iter().copied().for_each(&mut push);
        chunks.suffix().into_iter().for_each(&mut push);
        push(0);
        Self {
            entries,
            lead_padding: chunks.lead_padding() as u64,
        }
    }

    /// Returns the number of set bits before position `n`, for `n <= len`.
    #[expect(
        clippy::inline_always,
        reason = "must inline into the POPCNT-enabled caller to use the instruction"
    )]
    #[inline(always)]
    fn rank(&self, n: u64) -> u64 {
        let position = n + self.lead_padding;
        let (word, preceding) = self.entries[(position / 64) as usize];
        preceding + u64::from((word & ((1u64 << (position % 64)) - 1)).count_ones())
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

    /// Checks rank-based run filtering against a per-row reference over word-boundary run ends
    /// and a bit-offset mask.
    #[rstest]
    #[case(0, 200)]
    #[case(3, 200)]
    #[case(64, 128)]
    #[case(70, 129)]
    fn filter_run_end_primitive_matches_reference(
        #[case] offset: u64,
        #[case] length: u64,
    ) -> VortexResult<()> {
        let run_ends: Vec<u32> = [1u32, 2, 63, 64, 65, 127, 128, 129, 150, 192, 260, 300]
            .into_iter()
            .filter(|&end| u64::from(end) > offset)
            .collect();
        let mask_bits: Vec<bool> = (0..length).map(|i| (i * 7 + i / 5) % 3 == 0).collect();
        let length_usize = usize::try_from(length)?;
        // Start the mask at a non-zero bit offset in its backing buffer.
        let backing =
            BitBuffer::from_iter(std::iter::repeat_n(false, 5).chain(mask_bits.iter().copied()));
        let mask = BitBuffer::new_with_offset(backing.inner().clone(), length_usize, 5);

        let (ends, values_mask) = filter_run_end_primitive(
            &run_ends,
            offset,
            length,
            &mask,
            &mut SESSION.create_execution_ctx(),
        )?;

        let mut expected_ends = Vec::new();
        let mut expected_retained = Vec::new();
        let (mut start, mut selected) = (0usize, 0u32);
        for &end in &run_ends {
            let end = usize::try_from(u64::from(end) - offset)?.min(length_usize);
            let in_run = mask_bits[start..end].iter().filter(|&&bit| bit).count();
            selected += u32::try_from(in_run)?;
            if in_run > 0 {
                expected_ends.push(selected);
            }
            expected_retained.push(in_run > 0);
            start = end;
        }

        assert_eq!(ends.as_slice::<u32>(), expected_ends.as_slice());
        assert_eq!(values_mask, Mask::from_iter(expected_retained));
        Ok(())
    }
}
