// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::min;
use std::ops::AddAssign;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::Ordering;

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
use vortex_mask::MaskValues;

use crate::_benchmarking::RunEndFilterMode;
use crate::RunEnd;
use crate::compute::take::take_indices_unchecked;
const FILTER_TAKE_THRESHOLD: f64 = 0.1;
static FILTER_MODE_OVERRIDE: AtomicU8 = AtomicU8::new(RunEndFilterMode::Auto.as_u8());
static FILTER_MODE_OVERRIDE_LOCK: Mutex<()> = Mutex::new(());

pub(crate) fn override_run_end_filter_mode(mode: RunEndFilterMode) -> impl Drop {
    let lock = match FILTER_MODE_OVERRIDE_LOCK.lock() {
        Ok(lock) => lock,
        Err(err) => err.into_inner(),
    };
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
                            array.offset() as u64,
                            array.len() as u64,
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
    let true_count = mask_values.true_count();
    let runs_ratio = true_count as f64 / array.ends().len() as f64;

    if runs_ratio < FILTER_TAKE_THRESHOLD || true_count < 25 {
        FilterPath::Take
    } else {
        FilterPath::Encoded
    }
}

// Code adapted from apache arrow-rs https://github.com/apache/arrow-rs/blob/b1f5c250ebb6c1252b4e7c51d15b8e77f4c361fa/arrow-select/src/filter.rs#L425
fn filter_run_end_primitive<R: NativePType + AddAssign + From<bool> + AsPrimitive<u64>>(
    run_ends: &[R],
    offset: u64,
    length: u64,
    mask: &BitBuffer,
) -> VortexResult<(PrimitiveArray, Mask)> {
    let mut new_run_ends = buffer_mut![R::zero(); run_ends.len()];

    let mut start = 0u64;
    let mut j = 0;
    let mut count = R::zero();

    let new_mask: Mask = BitBuffer::collect_bool(run_ends.len(), |i| {
        let mut keep = false;
        let end = min(run_ends[i].as_() - offset, length);

        // Safety: predicate must be the same length as the array the ends have been taken from
        for pred in (start..end).map(|i| unsafe {
            mask.value_unchecked(i.try_into().vortex_expect("index must fit in usize"))
        }) {
            count += <R as From<bool>>::from(pred);
            keep |= pred
        }
        // this is to avoid branching
        new_run_ends[j] = count;
        j += keep as usize;

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
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use crate::RunEnd;
    use crate::RunEndArray;

    fn ree_array() -> RunEndArray {
        RunEnd::encode(PrimitiveArray::from_iter([1, 1, 1, 4, 4, 4, 2, 2, 5, 5, 5, 5]).into_array())
            .unwrap()
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
}
