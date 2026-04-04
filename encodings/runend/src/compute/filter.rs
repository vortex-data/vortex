// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::min;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::Ordering;

use num_traits::AsPrimitive;
use num_traits::NumCast;
use parking_lot::Mutex;
use parking_lot::MutexGuard;
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
use vortex_mask::MaskValues;

use crate::_benchmarking::RunEndFilterMode;
use crate::RunEnd;
use crate::compute::take::take_indices_unchecked;

const FILTER_TAKE_MIN_TRUE_COUNT: usize = 25;
const FILTER_ENCODED_DENSITY_SHIFT: usize = 3;
const FILTER_ENCODED_MIN_TRUES_PER_RUN: usize = 32;
const FILTER_ENCODED_MAX_SLICE_COUNT: usize = 32;
const FILTER_ENCODED_MIN_AVG_SLICE_LEN: usize = 256;

static FILTER_MODE_OVERRIDE: AtomicU8 = AtomicU8::new(RunEndFilterMode::Auto.as_u8());
static FILTER_MODE_OVERRIDE_LOCK: Mutex<()> = Mutex::new(());

pub(crate) fn override_run_end_filter_mode(mode: RunEndFilterMode) -> impl Drop {
    let lock = FILTER_MODE_OVERRIDE_LOCK.lock();
    let previous_mode = current_filter_mode();
    FILTER_MODE_OVERRIDE.store(mode.as_u8(), Ordering::SeqCst);

    RunEndFilterModeGuard {
        previous_mode,
        _lock: lock,
    }
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

        match select_filter_path(array, mask_values) {
            FilterPath::Take => Ok(Some(take_indices_unchecked(
                &array,
                mask_values.indices(),
                &Validity::NonNullable,
            )?)),
            FilterPath::Encoded => {
                let primitive_run_ends = array.ends().clone().execute::<PrimitiveArray>(ctx)?;
                let (run_ends, values_mask) =
                    match_each_unsigned_integer_ptype!(primitive_run_ends.ptype(), |P| {
                        filter_run_end_primitive(
                            primitive_run_ends.as_slice::<P>(),
                            array.offset(),
                            array.len(),
                            mask_values.bit_buffer(),
                        )?
                    });
                let values = array.values().filter(values_mask)?;

                // SAFETY: guaranteed by implementation of filter_run_end_primitive
                unsafe {
                    Ok(Some(
                        RunEnd::new_unchecked(
                            run_ends.into_array(),
                            values,
                            0,
                            mask_values.true_count(),
                        )
                        .into_array(),
                    ))
                }
            }
        }
    }
}

impl RunEndFilterMode {
    const fn as_u8(self) -> u8 {
        match self {
            Self::Auto => 0,
            Self::Take => 1,
            Self::Encoded => 2,
        }
    }

    const fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Take,
            2 => Self::Encoded,
            _ => Self::Auto,
        }
    }
}

struct RunEndFilterModeGuard {
    previous_mode: RunEndFilterMode,
    _lock: MutexGuard<'static, ()>,
}

impl Drop for RunEndFilterModeGuard {
    fn drop(&mut self) {
        FILTER_MODE_OVERRIDE.store(self.previous_mode.as_u8(), Ordering::SeqCst);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FilterPath {
    Take,
    Encoded,
}

fn current_filter_mode() -> RunEndFilterMode {
    RunEndFilterMode::from_u8(FILTER_MODE_OVERRIDE.load(Ordering::SeqCst))
}

fn select_filter_path(array: ArrayView<'_, RunEnd>, mask_values: &MaskValues) -> FilterPath {
    match current_filter_mode() {
        RunEndFilterMode::Auto => auto_filter_path(array, mask_values),
        RunEndFilterMode::Take => FilterPath::Take,
        RunEndFilterMode::Encoded => FilterPath::Encoded,
    }
}

fn auto_filter_path(array: ArrayView<'_, RunEnd>, mask_values: &MaskValues) -> FilterPath {
    let len = array.len();
    let run_count = array.ends().len();
    let true_count = mask_values.true_count();
    let slice_count = mask_values.slices().len();
    let average_slice_len = true_count.div_ceil(slice_count);

    if true_count < FILTER_TAKE_MIN_TRUE_COUNT {
        return FilterPath::Take;
    }

    let dense_selection = true_count.saturating_mul(1 << FILTER_ENCODED_DENSITY_SHIFT) >= len;
    let localized_selection = true_count
        >= run_count.saturating_mul(FILTER_ENCODED_MIN_TRUES_PER_RUN)
        && slice_count <= FILTER_ENCODED_MAX_SLICE_COUNT
        && average_slice_len >= FILTER_ENCODED_MIN_AVG_SLICE_LEN;

    if dense_selection || localized_selection {
        FilterPath::Encoded
    } else {
        FilterPath::Take
    }
}

fn filter_run_end_primitive<R: NativePType + AsPrimitive<usize>>(
    run_ends: &[R],
    offset: usize,
    length: usize,
    mask: &BitBuffer,
) -> VortexResult<(PrimitiveArray, Mask)> {
    let mut new_run_ends = buffer_mut![R::zero(); run_ends.len()];

    let mut start = 0usize;
    let mut j = 0;
    let mut count = 0u64;

    let new_mask: Mask = BitBuffer::collect_bool(run_ends.len(), |i| {
        let end = min(run_ends[i].as_() - offset, length);
        let run_true_count = mask.slice(start..end).true_count() as u64;
        let keep = run_true_count != 0;

        count += run_true_count;
        if keep {
            new_run_ends[j] =
                NumCast::from(count).vortex_expect("filtered run end count must fit in run-end type");
            j += 1;
        }

        start = end;
        keep
    })
    .into();

    new_run_ends.truncate(j);
    Ok((
        PrimitiveArray::new(new_run_ends, Validity::NonNullable),
        new_mask,
    ))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::cast_possible_truncation)]

    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use super::FilterPath;
    use super::override_run_end_filter_mode;
    use super::select_filter_path;
    use crate::_benchmarking::RunEndFilterMode;
    use crate::RunEnd;
    use crate::RunEndArray;

    fn ree_array() -> RunEndArray {
        RunEnd::encode(PrimitiveArray::from_iter([1, 1, 1, 4, 4, 4, 2, 2, 5, 5, 5, 5]).into_array())
            .unwrap()
    }

    fn run_end_fixture(run_length: usize, len: usize) -> ArrayRef {
        let run_count = len.div_ceil(run_length);
        let ends = (0..run_count)
            .map(|run_idx| ((run_idx + 1) * run_length).min(len) as u32)
            .collect::<Buffer<_>>()
            .into_array();
        let values =
            PrimitiveArray::from_iter((0..run_count).map(|run_idx| run_idx as i32)).into_array();

        RunEnd::new(ends, values).into_array()
    }

    fn run_end_offset_fixture(
        run_length: usize,
        total_len: usize,
        offset: usize,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        let run_count = total_len.div_ceil(run_length);
        let ends = (0..run_count)
            .map(|run_idx| ((run_idx + 1) * run_length).min(total_len) as u32)
            .collect::<Buffer<_>>()
            .into_array();
        let values =
            PrimitiveArray::from_iter((0..run_count).map(|run_idx| run_idx as i32)).into_array();

        Ok(RunEnd::try_new_offset_length(ends, values, offset, len)?.into_array())
    }

    fn sparse_random_mask(len: usize, true_count: usize) -> Mask {
        let mut indices = (0..true_count)
            .map(|idx| (idx * 7_919) % len)
            .collect::<Vec<_>>();
        indices.sort_unstable();

        Mask::from_indices(len, indices)
    }

    fn sparse_clustered_mask(len: usize) -> Mask {
        Mask::from_slices(len, vec![(1_024, 1_536), (8_192, 8_704)])
    }

    fn sparse_clustered_mask_for_slice(len: usize) -> Mask {
        Mask::from_slices(len, vec![(1_024, 1_536), (6_144, 6_656)])
    }

    fn filter_path(array: &ArrayRef, mask: &Mask) -> FilterPath {
        select_filter_path(
            array.as_::<RunEnd>(),
            mask.values()
                .expect("heuristic tests require a partial filter mask"),
        )
    }

    fn filter_with_mode(array: &ArrayRef, mask: Mask, mode: RunEndFilterMode) -> VortexResult<ArrayRef> {
        let _guard = override_run_end_filter_mode(mode);
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        array.filter(mask)?.execute::<ArrayRef>(&mut ctx)
    }

    fn assert_encoded_filter_matches_take(array: &ArrayRef, mask: Mask) -> VortexResult<()> {
        let take_filtered = filter_with_mode(array, mask.clone(), RunEndFilterMode::Take)?;
        let encoded_filtered = filter_with_mode(array, mask, RunEndFilterMode::Encoded)?;

        assert_arrays_eq!(encoded_filtered, take_filtered);
        Ok(())
    }

    #[test]
    fn filter_sliced_run_end() -> VortexResult<()> {
        let arr = ree_array().slice(2..7).unwrap();
        let filtered = arr.filter(Mask::from_iter([true, false, false, true, true]))?;

        assert_arrays_eq!(
            filtered,
            RunEnd::new(
                PrimitiveArray::from_iter([1u8, 2, 3]).into_array(),
                PrimitiveArray::from_iter([1i32, 4, 2]).into_array()
            )
        );
        Ok(())
    }

    #[test]
    fn heuristic_prefers_take_for_sparse_random_mask() -> VortexResult<()> {
        let array = run_end_fixture(1_024, 16_384);
        let mask = sparse_random_mask(array.len(), 1_024);

        assert_eq!(filter_path(&array, &mask), FilterPath::Take);
        Ok(())
    }

    #[test]
    fn heuristic_prefers_encoded_for_sparse_clustered_mask() -> VortexResult<()> {
        let array = run_end_fixture(1_024, 16_384);
        let mask = sparse_clustered_mask(array.len());

        assert_eq!(filter_path(&array, &mask), FilterPath::Encoded);
        Ok(())
    }

    #[test]
    fn heuristic_prefers_take_for_sparse_random_mask_on_slice() -> VortexResult<()> {
        let array = run_end_offset_fixture(1_024, 16_384, 1_024, 14_336)?;
        let mask = sparse_random_mask(array.len(), 1_024);

        assert_eq!(filter_path(&array, &mask), FilterPath::Take);
        Ok(())
    }

    #[test]
    fn heuristic_prefers_encoded_for_sparse_clustered_mask_on_slice() -> VortexResult<()> {
        let array = run_end_offset_fixture(1_024, 16_384, 1_024, 14_336)?;
        let mask = sparse_clustered_mask_for_slice(array.len());

        assert_eq!(filter_path(&array, &mask), FilterPath::Encoded);
        Ok(())
    }

    #[test]
    fn encoded_filter_matches_take_on_partial_word_boundaries() -> VortexResult<()> {
        let array = run_end_fixture(65, 260);
        let mask = Mask::from_slices(array.len(), vec![(3, 64), (67, 129), (133, 194), (197, 259)]);

        assert_encoded_filter_matches_take(&array, mask)
    }

    #[test]
    fn encoded_filter_matches_take_on_clustered_masks() -> VortexResult<()> {
        let array = run_end_fixture(1_024, 16_384);
        let mask = Mask::from_slices(array.len(), vec![(13, 513), (4_109, 4_733), (9_001, 10_129)]);

        assert_encoded_filter_matches_take(&array, mask)
    }

    #[test]
    fn encoded_filter_matches_take_on_very_short_runs() -> VortexResult<()> {
        let array = run_end_fixture(1, 64);
        let mask = Mask::from_slices(array.len(), vec![(1, 3), (5, 8), (13, 14), (21, 25), (34, 35)]);

        assert_encoded_filter_matches_take(&array, mask)
    }
}
