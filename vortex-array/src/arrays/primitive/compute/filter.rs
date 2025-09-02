// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::{Buffer, BufferMut};
use vortex_dtype::match_each_native_ptype;
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::{Mask, MaskIter};

use crate::arrays::PrimitiveVTable;
use crate::arrays::primitive::PrimitiveArray;
use crate::compute::{FilterKernel, FilterKernelAdapter};
use crate::vtable::ValidityHelper;
use crate::{ArrayRef, IntoArray, register_kernel};

// This is modeled after the constant with the equivalent name in arrow-rs.
const FILTER_SLICES_SELECTIVITY_THRESHOLD: f64 = 0.8;

impl FilterKernel for PrimitiveVTable {
    fn filter(&self, array: &PrimitiveArray, mask: &Mask) -> VortexResult<ArrayRef> {
        let validity = array.validity().filter(mask)?;

        let mask_values = mask
            .values()
            .vortex_expect("AllTrue and AllFalse are handled by filter fn");

        match mask_values.threshold_iter(FILTER_SLICES_SELECTIVITY_THRESHOLD) {
            MaskIter::Indices(indices) => {
                match_each_native_ptype!(array.ptype(), |T| {
                    let values =
                        filter_primitive_indices(array.as_slice::<T>(), indices.iter().copied());
                    Ok(PrimitiveArray::new(values, validity).into_array())
                })
            }
            MaskIter::Slices(slices) => {
                match_each_native_ptype!(array.ptype(), |T| {
                    let values = filter_primitive_slices(
                        array.as_slice::<T>(),
                        mask.true_count(),
                        slices.iter().copied(),
                    );
                    Ok(PrimitiveArray::new(values, validity).into_array())
                })
            }
        }
    }
}

register_kernel!(FilterKernelAdapter(PrimitiveVTable).lift());

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
#[allow(clippy::cast_possible_truncation)]
mod test {
    use itertools::Itertools;
    use rstest::rstest;
    use vortex_mask::Mask;

    use crate::arrays::primitive::PrimitiveArray;
    use crate::canonical::ToCanonical;
    use crate::compute::conformance::filter::{LARGE_SIZE, MEDIUM_SIZE, test_filter_conformance};
    use crate::compute::filter;

    #[test]
    fn filter_run_variant_mixed_test() {
        let mask = [true, true, false, true, true, true, false, true];
        let arr = PrimitiveArray::from_iter([1u32, 24, 54, 2, 3, 2, 3, 2]);

        let filtered = filter(arr.as_ref(), &Mask::from_iter(mask))
            .unwrap()
            .to_primitive();
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

    #[rstest]
    #[case(PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]))]
    #[case(PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4), None]))]
    #[case(PrimitiveArray::from_iter([42u64]))]
    #[case(PrimitiveArray::from_iter(0..MEDIUM_SIZE as i32))]
    #[case(PrimitiveArray::from_option_iter((0..MEDIUM_SIZE).map(|i| if i % 3 == 0 { None } else { Some(i as i64) })))]
    #[case(PrimitiveArray::from_iter(0..LARGE_SIZE as u32))]
    #[case(PrimitiveArray::from_iter([0.1f32, 0.2, 0.3, 0.4, 0.5]))]
    #[case(PrimitiveArray::from_option_iter([Some(1.1f64), None, Some(2.2), Some(3.3), None]))]
    fn test_filter_primitive_conformance(#[case] array: PrimitiveArray) {
        test_filter_conformance(array.as_ref());
    }
}
