// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::{BoolArray, BoolVTable};
use crate::compute::{MaskKernel, MaskKernelAdapter};
use crate::vtable::ValidityHelper;
use crate::{ArrayRef, IntoArray, register_kernel};

impl MaskKernel for BoolVTable {
    fn mask(&self, array: &BoolArray, mask: &Mask) -> VortexResult<ArrayRef> {
        Ok(
            BoolArray::from_bit_buffer(array.bit_buffer().clone(), array.validity().mask(mask))
                .into_array(),
        )
    }
}

register_kernel!(MaskKernelAdapter(BoolVTable).lift());

#[cfg(test)]
mod test {
    use rstest::rstest;

    use crate::arrays::BoolArray;
    use crate::compute::conformance::mask::test_mask_conformance;

    #[rstest]
    #[case(BoolArray::from_iter([true, false, true, true, false]))]
    #[case(BoolArray::from_iter([Some(true), None, Some(false), Some(true), None]))]
    #[case(BoolArray::from_iter([true]))]
    #[case(BoolArray::from_iter([false, false]))]
    #[case(BoolArray::from_iter((0..100).map(|i| i % 2 == 0)))]
    fn test_mask_bool_conformance(#[case] array: BoolArray) {
        test_mask_conformance(array.as_ref());
    }
}
