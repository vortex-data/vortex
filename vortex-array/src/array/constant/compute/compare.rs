use vortex_error::VortexResult;

use crate::array::ConstantArray;
use crate::compute::{scalar_cmp, CompareFn, Operator};
use crate::{ArrayData, ArrayLen, IntoArrayData};

impl CompareFn for ConstantArray {
    fn compare(&self, other: &ArrayData, operator: Operator) -> VortexResult<Option<ArrayData>> {
        // We only support comparing a constant array to another constant array.
        // For all other encodings, we assume the constant is on the RHS.
        if let Some(const_scalar) = other.as_constant() {
            let lhs = self.owned_scalar();
            let scalar = scalar_cmp(&lhs, &const_scalar, operator);
            return Ok(Some(ConstantArray::new(scalar, self.len()).into_array()));
        }

        Ok(None)
    }
}
