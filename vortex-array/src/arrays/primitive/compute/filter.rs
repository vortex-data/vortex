// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::PrimitiveVTable;
use crate::arrays::filter::FILTER_SLICES_SELECTIVITY_THRESHOLD;
use crate::arrays::filter::filter_slice;
use crate::arrays::primitive::PrimitiveArray;
use crate::compute::FilterKernel;
use crate::compute::FilterKernelAdapter;
use crate::register_kernel;
use crate::vtable::ValidityHelper;

impl FilterKernel for PrimitiveVTable {
    fn filter(&self, array: &PrimitiveArray, mask: &Mask) -> VortexResult<ArrayRef> {
        let validity = array.validity().filter(mask)?;

        match_each_native_ptype!(array.ptype(), |T| {
            let values = filter_slice(
                array.as_slice::<T>(),
                mask,
                FILTER_SLICES_SELECTIVITY_THRESHOLD,
            );
            Ok(PrimitiveArray::new(values, validity).into_array())
        })
    }
}

register_kernel!(FilterKernelAdapter(PrimitiveVTable).lift());

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod test {
    use itertools::Itertools;
    use rstest::rstest;
    use vortex_mask::Mask;

    use crate::arrays::primitive::PrimitiveArray;
    use crate::canonical::ToCanonical;
    use crate::compute::conformance::filter::LARGE_SIZE;
    use crate::compute::conformance::filter::MEDIUM_SIZE;
    use crate::compute::conformance::filter::test_filter_conformance;
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
