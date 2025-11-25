// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::Nullability;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::arrays::MaskedArray;
use crate::compute::MaskKernel;
use crate::compute::MaskKernelAdapter;
use crate::register_kernel;
use crate::validity::Validity;

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
