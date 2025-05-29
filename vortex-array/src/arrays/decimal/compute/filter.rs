use vortex_buffer::{Buffer, BufferMut};
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::{Mask, MaskIter};
use vortex_scalar::match_each_decimal_value_type;

use crate::arrays::{DecimalArray, DecimalVTable};
use crate::compute::{FilterKernel, FilterKernelAdapter};
use crate::vtable::ValidityHelper;
use crate::{ArrayRef, IntoArray, register_kernel};

const FILTER_SLICES_SELECTIVITY_THRESHOLD: f64 = 0.8;

impl FilterKernel for DecimalVTable {
    fn filter(&self, array: &DecimalArray, mask: &Mask) -> VortexResult<ArrayRef> {
        let validity = array.validity().filter(mask)?;

        let mask_values = mask
            .values()
            .vortex_expect("AllTrue and AllFalse are handled by filter fn");

        macro_rules! filter_by_indices {
            ($ty:ty, $array:expr, $indices:expr, $validity:expr) => {{
                let filtered = filter_primitive_indices::<$ty>(
                    $array.buffer().as_slice(),
                    $indices.iter().copied(),
                );
                Ok(DecimalArray::new(filtered, $array.decimal_dtype(), $validity).into_array())
            }};
        }

        macro_rules! filter_by_slices {
            ($ty:ty, $array:expr, $mask:expr, $slices:expr, $validity:expr) => {{
                let filtered = filter_primitive_slices::<$ty>(
                    $array.buffer().as_slice(),
                    $mask.true_count(),
                    $slices.iter().copied(),
                );
                Ok(DecimalArray::new(filtered, $array.decimal_dtype(), $validity).into_array())
            }};
        }

        match mask_values.threshold_iter(FILTER_SLICES_SELECTIVITY_THRESHOLD) {
            MaskIter::Indices(indices) => match_each_decimal_value_type!(array.values_type, |S| {
                filter_by_indices!(S, array, indices, validity)
            }),

            MaskIter::Slices(slices) => match_each_decimal_value_type!(array.values_type, |S| {
                filter_by_slices!(S, array, mask, slices, validity)
            }),
        }
    }
}

register_kernel!(FilterKernelAdapter(DecimalVTable).lift());

fn filter_primitive_indices<T: Copy>(
    values: &[T],
    indices: impl Iterator<Item = usize>,
) -> Buffer<T> {
    indices
        .map(|idx| *unsafe { values.get_unchecked(idx) })
        .collect()
}

fn filter_primitive_slices<T: Clone>(
    values: &[T],
    indices_len: usize,
    indices: impl Iterator<Item = (usize, usize)>,
) -> Buffer<T> {
    let mut output = BufferMut::with_capacity(indices_len);
    for (start, end) in indices {
        output.extend_from_slice(&values[start..end]);
    }
    output.freeze()
}
