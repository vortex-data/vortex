// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::arrays::MaskedArray;
use crate::compute::MaskReduce;
use crate::validity::Validity;

impl MaskReduce for ConstantVTable {
    fn mask(array: &ConstantArray, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        if array.scalar.is_null() {
            // Already all nulls, masking has no effect.
            return Ok(Some(array.to_array()));
        }
        Ok(Some(
            MaskedArray::try_new(array.to_array(), Validity::Array(mask.clone()))?.into_array(),
        ))
    }
}

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
