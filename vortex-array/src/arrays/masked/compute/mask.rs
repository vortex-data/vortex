// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask as MaskType;

use crate::arrays::{MaskedArray, MaskedVTable};
use crate::compute::{MaskKernel, MaskKernelAdapter};
use crate::vtable::ValidityHelper;
use crate::{ArrayRef, IntoArray, register_kernel};

impl MaskKernel for MaskedVTable {
    fn mask(&self, array: &MaskedArray, mask_arg: &MaskType) -> VortexResult<ArrayRef> {
        // Combine the mask with the existing validity
        // The child remains unchanged (no nulls), only validity is updated
        let combined_validity = array.validity().mask(mask_arg);

        Ok(MaskedArray::try_new(array.child.clone(), combined_validity)?.into_array())
    }
}

register_kernel!(MaskKernelAdapter(MaskedVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::IntoArray;
    use crate::arrays::{MaskedArray, PrimitiveArray};
    use crate::compute::conformance::mask::test_mask_conformance;
    use crate::validity::Validity;

    #[rstest]
    #[case(
        MaskedArray::try_new(
            PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]).into_array(),
            Validity::from_iter([true, true, false, true, false])
        ).unwrap()
    )]
    #[case(
        MaskedArray::try_new(
            PrimitiveArray::from_iter([10i32, 20, 30]).into_array(),
            Validity::AllValid
        ).unwrap()
    )]
    #[case(
        MaskedArray::try_new(
            PrimitiveArray::from_iter(0..100).into_array(),
            Validity::from_iter((0..100).map(|i| i % 3 != 0))
        ).unwrap()
    )]
    fn test_mask_masked_conformance(#[case] array: MaskedArray) {
        test_mask_conformance(array.as_ref());
    }
}
