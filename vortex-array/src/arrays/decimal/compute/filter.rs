use vortex_buffer::{Buffer, BufferMut};
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::{Mask, MaskIter};
use vortex_scalar::i256;

use crate::arrays::decimal::serde::DecimalValueType;
use crate::arrays::{DecimalArray, DecimalEncoding};
use crate::compute::{FilterKernelAdapter, FilterKernelImpl};
use crate::{Array, ArrayRef, register_kernel};

const FILTER_SLICES_SELECTIVITY_THRESHOLD: f64 = 0.8;

impl FilterKernelImpl for DecimalEncoding {
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
            MaskIter::Indices(indices) => match array.values_type {
                DecimalValueType::I128 => filter_by_indices!(i128, array, indices, validity),
                DecimalValueType::I256 => filter_by_indices!(i256, array, indices, validity),
            },
            MaskIter::Slices(slices) => match array.values_type {
                DecimalValueType::I128 => filter_by_slices!(i128, array, mask, slices, validity),
                DecimalValueType::I256 => filter_by_slices!(i256, array, mask, slices, validity),
            },
        }
    }
}

register_kernel!(FilterKernelAdapter(DecimalEncoding).lift());

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
