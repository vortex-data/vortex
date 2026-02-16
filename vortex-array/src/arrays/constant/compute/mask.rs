// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::arrays::ScalarFnArrayExt;
use crate::compute::MaskReduce;
use crate::expr::EmptyOptions;
use crate::expr::mask::Mask as MaskExpr;

impl MaskReduce for ConstantVTable {
    fn mask(array: &ConstantArray, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        if array.scalar.is_null() {
            // Already all nulls, masking has no effect.
            return Ok(Some(array.to_array()));
        }
        Ok(Some(MaskExpr.try_new_array(
            array.len(),
            EmptyOptions,
            [array.to_array(), mask.clone()],
        )?))
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
