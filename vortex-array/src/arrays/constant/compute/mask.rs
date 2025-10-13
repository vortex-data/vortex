// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::Nullability;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::{ConstantArray, ConstantVTable, MaskedArray};
use crate::compute::{MaskKernel, MaskKernelAdapter};
use crate::validity::Validity;
use crate::{ArrayRef, IntoArray, register_kernel};

impl MaskKernel for ConstantVTable {
    fn mask(&self, array: &ConstantArray, mask: &Mask) -> VortexResult<ArrayRef> {
        MaskedArray::try_new(
            array.to_array(),
            Validity::from_mask(!mask, Nullability::Nullable),
        )
        .map(|a| a.into_array())
    }
}

register_kernel!(MaskKernelAdapter(ConstantVTable).lift());

#[cfg(test)]
mod test {
    use crate::arrays::ConstantArray;
    use crate::compute::conformance::mask::test_mask_conformance;

    #[test]
    fn test_mask_constant() {
        let array = ConstantArray::new(std::f64::consts::PI, 15);
        test_mask_conformance(array.as_ref());
    }
}
