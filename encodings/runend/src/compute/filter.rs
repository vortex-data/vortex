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
use vortex_array::dtype::UnsignedPType;
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

/// Uses one sequential bitmap cursor when the average run length is at most this value.
///
/// The general range popcount performs less work for long runs. For short runs, its repeated
/// alignment and range setup costs more than a cursor that keeps the current bitmap word loaded.
const SEQUENTIAL_MASK_SCAN_MAX_AVERAGE_RUN_LENGTH: u64 = 64;

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
pub fn filter_run_end_primitive<R: UnsignedPType + AddAssign + From<bool> + AsPrimitive<u64>>(
    run_ends: &[R],
    offset: u64,
    length: u64,
    mask: &BitBuffer,
) -> VortexResult<(PrimitiveArray, Mask)> {
    if length <= (run_ends.len() as u64).saturating_mul(SEQUENTIAL_MASK_SCAN_MAX_AVERAGE_RUN_LENGTH)
    {
        return Ok(filter_run_end_sequential(run_ends, offset, length, mask));
    }

    filter_run_end_ranges(run_ends, offset, length, mask)
}

#[doc(hidden)]
pub fn filter_run_end_ranges<R: UnsignedPType + AddAssign + From<bool> + AsPrimitive<u64>>(
    run_ends: &[R],
    offset: u64,
    length: u64,
    mask: &BitBuffer,
) -> VortexResult<(PrimitiveArray, Mask)> {
    let mut filtered_run_ends = buffer_mut![R::zero(); run_ends.len()];

    let mut run_start = 0u64;
    let mut retained_run_count = 0;
    let mut filtered_end = R::zero();

    let values_mask: Mask = BitBuffer::collect_bool(run_ends.len(), |run_idx| {
        let absolute_run_end: u64 = run_ends[run_idx].as_();
        let run_end = min(absolute_run_end - offset, length);

        // Bulk popcount is SIMD-capable and avoids per-bit reads. The input contract and clamp prove
        // `run_start_idx <= run_end_idx <= mask.len()`.
        let run_start_idx = run_start
            .try_into()
            .vortex_expect("run start index must fit in usize");
        let run_end_idx = run_end
            .try_into()
            .vortex_expect("run end index must fit in usize");
        let selected_in_run = mask.count_range(run_start_idx, run_end_idx);
        filtered_end += <R as NumCast>::from(selected_in_run)
            .vortex_expect("run popcount must fit in run-end native type");
        let retain_run = selected_in_run > 0;

        // Always write the current end, then advance only for a retained run. This keeps the loop
        // branchless.
        filtered_run_ends[retained_run_count] = filtered_end;
        retained_run_count += retain_run as usize;

        run_start = run_end;
        retain_run
    })
    .into();

    filtered_run_ends.truncate(retained_run_count);

    Ok((
        PrimitiveArray::new(filtered_run_ends, Validity::NonNullable),
        values_mask,
    ))
}

/// Recomputes run ends with one sequential cursor over the filter bitmap.
///
/// The mask must contain `length` bits. The first run end can equal `offset` for a sliced array.
/// Later run ends must increase and cover the filtered range.
///
/// # Panics
///
/// Panics if the mask length differs from `length`, or if the run ends violate the documented
/// order.
#[doc(hidden)]
pub fn filter_run_end_sequential<R: UnsignedPType + AddAssign + From<bool> + AsPrimitive<u64>>(
    run_ends: &[R],
    offset: u64,
    length: u64,
    mask: &BitBuffer,
) -> (PrimitiveArray, Mask) {
    let mut filtered_run_ends = buffer_mut![R::zero(); run_ends.len()];
    let chunks = mask.chunks();
    let mask_length = usize::try_from(length).vortex_expect("mask length must fit in usize");
    assert_eq!(
        mask.len(),
        mask_length,
        "filter mask length must equal the run-end array length"
    );
    let mut mask_cursor = MaskCountCursor::new(chunks.iter_padded(), mask_length);
    let mut retained_run_count = 0;
    let mut filtered_end = R::zero();
    let mut previous_absolute_run_end = offset;

    let values_mask = BitBuffer::collect_bool(run_ends.len(), |run_index| {
        let absolute_run_end = run_ends[run_index].as_();
        if run_index == 0 {
            assert!(
                absolute_run_end >= offset,
                "first run end {absolute_run_end} is before offset {offset}"
            );
        } else {
            assert!(
                absolute_run_end > previous_absolute_run_end,
                "run end {absolute_run_end} does not follow {previous_absolute_run_end}"
            );
        }
        previous_absolute_run_end = absolute_run_end;
        let run_end = min(absolute_run_end - offset, length)
            .try_into()
            .vortex_expect("run end must fit in usize");
        let selected_in_run = mask_cursor.count_to(run_end);
        filtered_end += <R as NumCast>::from(selected_in_run)
            .vortex_expect("run popcount must fit in run-end native type");
        let retain_run = selected_in_run > 0;
        filtered_run_ends[retained_run_count] = filtered_end;
        retained_run_count += retain_run as usize;
        retain_run
    })
    .into();
    assert_eq!(
        mask_cursor.position, mask_length,
        "run ends must cover the filtered range"
    );

    filtered_run_ends.truncate(retained_run_count);
    (
        PrimitiveArray::new(filtered_run_ends, Validity::NonNullable),
        values_mask,
    )
}

struct MaskCountCursor<I> {
    words: I,
    current_word: u64,
    position: usize,
    length: usize,
}

impl<I: Iterator<Item = u64>> MaskCountCursor<I> {
    fn new(mut words: I, length: usize) -> Self {
        let current_word = if length == 0 {
            0
        } else {
            words.next().vortex_expect("mask word must exist")
        };
        Self {
            words,
            current_word,
            position: 0,
            length,
        }
    }

    fn count_to(&mut self, end: usize) -> usize {
        assert!(end >= self.position);
        assert!(end <= self.length);

        let mut selected = 0;
        while self.position < end {
            let bit_in_word = self.position % 64;
            let bits_to_read = (end - self.position).min(64 - bit_in_word);
            let bit_mask = if bits_to_read == 64 {
                u64::MAX
            } else {
                (1u64 << bits_to_read) - 1
            };
            selected += ((self.current_word >> bit_in_word) & bit_mask).count_ones() as usize;
            self.position += bits_to_read;
            if self.position.is_multiple_of(64) && self.position < self.length {
                self.current_word = self.words.next().vortex_expect("mask word must exist");
            }
        }
        selected
    }
}

#[cfg(test)]
mod tests {
    use std::ops::AddAssign;

    use num_traits::AsPrimitive;
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_array::dtype::UnsignedPType;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::Buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use super::filter_run_end_ranges;
    use super::filter_run_end_sequential;
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

    #[rstest]
    #[case::one_row_runs(0, 1, 0)]
    #[case::short_runs_with_bitmap_offset(1, 4, 0)]
    #[case::threshold_runs_with_slice(7, 64, 31)]
    #[case::long_runs_with_slice(5, 256, 127)]
    fn sequential_scan_matches_range_reference(
        #[case] bitmap_offset: usize,
        #[case] run_length: usize,
        #[case] array_offset: usize,
    ) -> VortexResult<()> {
        let length = 1_003;
        let source_length = array_offset + length;
        let run_ends = (0..source_length.div_ceil(run_length))
            .map(|run_index| {
                u32::try_from(((run_index + 1) * run_length).min(source_length))
                    .vortex_expect("test run end must fit in u32")
            })
            .collect::<Vec<_>>();

        assert_scan_matches_reference(&run_ends, array_offset, bitmap_offset, length)
    }

    #[test]
    fn irregular_sequential_scan_matches_range_reference() -> VortexResult<()> {
        assert_scan_matches_reference(&[7u32, 71, 72, 145, 146, 500, 501, 1_008], 5, 3, 1_003)
    }

    #[test]
    fn sequential_scan_accepts_empty_leading_run() -> VortexResult<()> {
        assert_scan_matches_reference(&[64u32, 128, 192], 64, 0, 128)
    }

    #[test]
    fn sequential_scan_supports_all_run_end_widths() -> VortexResult<()> {
        assert_scan_matches_reference(&[7u8, 71, 72, 145, 146, 200], 5, 3, 195)?;
        assert_scan_matches_reference(&[7u16, 71, 72, 145, 146, 200], 5, 3, 195)?;
        assert_scan_matches_reference(&[7u32, 71, 72, 145, 146, 200], 5, 3, 195)?;
        assert_scan_matches_reference(&[7u64, 71, 72, 145, 146, 200], 5, 3, 195)
    }

    fn assert_scan_matches_reference<R>(
        run_ends: &[R],
        array_offset: usize,
        bitmap_offset: usize,
        length: usize,
    ) -> VortexResult<()>
    where
        R: UnsignedPType + AddAssign + From<bool> + AsPrimitive<u64>,
    {
        let backing_mask = BitBuffer::collect_bool(bitmap_offset + length, |index| {
            index >= bitmap_offset && !(index - bitmap_offset).is_multiple_of(5)
        });
        let mask = backing_mask.slice(bitmap_offset..bitmap_offset + length);

        let (actual_ends, actual_values_mask) =
            filter_run_end_sequential(run_ends, array_offset as u64, length as u64, &mask);
        let (expected_ends, expected_values_mask) =
            filter_run_end_ranges(run_ends, array_offset as u64, length as u64, &mask)?;

        assert_eq!(actual_ends.as_slice::<R>(), expected_ends.as_slice::<R>());
        assert_eq!(actual_values_mask, expected_values_mask);
        Ok(())
    }

    #[rstest]
    #[case::short_runs(4, false, false)]
    #[case::long_nullable_sliced_runs(256, true, true)]
    fn filter_conformance(
        #[case] run_length: usize,
        #[case] nullable: bool,
        #[case] sliced: bool,
    ) -> VortexResult<()> {
        let leading_slice = if sliced { run_length / 2 } else { 0 };
        let length = 1_024;
        let source_length = leading_slice + length;
        let run_count = source_length.div_ceil(run_length);
        let ends = (0..run_count)
            .map(|run_index| {
                u32::try_from(((run_index + 1) * run_length).min(source_length))
                    .vortex_expect("test run end must fit in u32")
            })
            .collect::<Buffer<_>>()
            .into_array();
        let values = if nullable {
            PrimitiveArray::from_option_iter((0..run_count).map(|run_index| {
                (!run_index.is_multiple_of(7))
                    .then(|| i32::try_from(run_index).vortex_expect("test value must fit in i32"))
            }))
            .into_array()
        } else {
            PrimitiveArray::from_iter((0..run_count).map(|run_index| {
                i32::try_from(run_index).vortex_expect("test value must fit in i32")
            }))
            .into_array()
        };
        let array = if sliced {
            RunEnd::try_new_offset_length(
                ends,
                values,
                leading_slice,
                length,
                &mut SESSION.create_execution_ctx(),
            )?
        } else {
            RunEnd::try_new(ends, values, &mut SESSION.create_execution_ctx())?
        };

        test_filter_conformance(&array.into_array(), &mut SESSION.create_execution_ctx());
        Ok(())
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
}
