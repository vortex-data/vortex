use vortex_buffer::{Buffer, BufferMut};
use vortex_dtype::match_each_native_ptype;
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::{Mask, MaskIter};

use crate::arrays::primitive::PrimitiveArray;
use crate::arrays::PrimitiveEncoding;
use crate::compute::FilterFn;
use crate::variants::PrimitiveArrayTrait;
use crate::{Array, ArrayRef};

// This is modeled after the constant with the equivalent name in arrow-rs.
const FILTER_SLICES_SELECTIVITY_THRESHOLD: f64 = 0.8;

impl FilterFn<&PrimitiveArray> for PrimitiveEncoding {
    fn filter(&self, array: &PrimitiveArray, mask: &Mask) -> VortexResult<ArrayRef> {
        let validity = array.validity().filter(mask)?;

        let mask_values = mask
            .values()
            .vortex_expect("AllTrue and AllFalse are handled by filter fn");

        match mask_values.threshold_iter(FILTER_SLICES_SELECTIVITY_THRESHOLD) {
            MaskIter::Indices(indices) => {
                match_each_native_ptype!(array.ptype(), |$T| {
                    let values = filter_primitive_indices(array.as_slice::<$T>(), indices.iter().copied());
                    Ok(PrimitiveArray::new(values, validity).into_array())
                })
            }
            MaskIter::Slices(slices) => {
                match_each_native_ptype!(array.ptype(), |$T| {
                    let values = filter_primitive_slices(array.as_slice::<$T>(), mask.true_count(), slices.iter().copied());
                    Ok(PrimitiveArray::new(values, validity).into_array())
                })
            }
        }
    }
}

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

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use vortex_mask::Mask;

    use crate::array::Array;
    use crate::arrays::primitive::PrimitiveArray;
    use crate::canonical::ToCanonical;
    use crate::compute::filter;

    #[test]
    fn filter_run_variant_mixed_test() {
        let mask = [true, true, false, true, true, true, false, true];
        let arr = PrimitiveArray::from_iter([1u32, 24, 54, 2, 3, 2, 3, 2]);

        let filtered = filter(&arr, &Mask::from_iter(mask))
            .unwrap()
            .to_primitive()
            .unwrap();
        assert_eq!(
            filtered.len(),
            mask.iter().filter(|x| **x).collect_vec().len()
        );

        let rust_arr = arr.as_slice::<u32>();
        assert_eq!(
            filtered.as_slice::<u32>().to_vec(),
            mask.iter()
                .enumerate()
                .filter(|(_idx, b)| **b)
                .map(|m| rust_arr[m.0])
                .collect_vec()
        )
    }
}
