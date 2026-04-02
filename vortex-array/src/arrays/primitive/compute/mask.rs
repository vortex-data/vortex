// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::scalar_fn::fns::mask::MaskReduce;
use crate::validity::Validity;

impl MaskReduce for Primitive {
    fn mask(array: ArrayView<'_, Primitive>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        // SAFETY: validity and data buffer still have same length
        Ok(Some(unsafe {
            PrimitiveArray::new_unchecked_from_handle(
                array.buffer_handle().clone(),
                array.ptype(),
                array.validity().and(Validity::Array(mask.clone()))?,
            )
            .into_array()
        }))
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;

    use crate::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::compute::conformance::mask::test_mask_conformance;

    #[rstest]
    #[case(PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]))]
    #[case(PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4), None]))]
    #[case(PrimitiveArray::from_iter([42u64]))]
    #[case(PrimitiveArray::from_iter(0..100i32))]
    #[case(PrimitiveArray::from_option_iter((0..100).map(|i| if i % 5 == 0 { None } else { Some(i as i64) })))]
    #[case(PrimitiveArray::from_iter([0.1f32, 0.2, 0.3, 0.4, 0.5]))]
    #[case(PrimitiveArray::from_option_iter([Some(1.1f64), None, Some(2.2), Some(3.3), None]))]
    fn test_mask_primitive_conformance(#[case] array: PrimitiveArray) {
        test_mask_conformance(&array.into_array());
    }
}
