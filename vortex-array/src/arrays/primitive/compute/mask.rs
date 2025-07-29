// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::PrimitiveVTable;
use crate::arrays::primitive::PrimitiveArray;
use crate::compute::{MaskKernel, MaskKernelAdapter};
use crate::vtable::ValidityHelper;
use crate::{ArrayRef, IntoArray, register_kernel};

impl MaskKernel for PrimitiveVTable {
    fn mask(&self, array: &PrimitiveArray, mask: &Mask) -> VortexResult<ArrayRef> {
        let validity = array.validity().mask(mask)?;
        Ok(
            PrimitiveArray::from_byte_buffer(array.byte_buffer().clone(), array.ptype(), validity)
                .into_array(),
        )
    }
}

register_kernel!(MaskKernelAdapter(PrimitiveVTable).lift());

#[cfg(test)]
mod test {
    use rstest::rstest;

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
        test_mask_conformance(array.as_ref());
    }
}
