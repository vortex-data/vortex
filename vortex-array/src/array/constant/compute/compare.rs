use vortex_error::VortexResult;

use crate::array::{ConstantArray, ConstantEncoding};
use crate::compute::{scalar_cmp, CompareFn, Operator};
use crate::{ArrayData, ArrayLen, IntoArrayData};

impl CompareFn<ConstantArray> for ConstantEncoding {
    fn compare(
        &self,
        lhs: &ConstantArray,
        rhs: &ArrayData,
        operator: Operator,
    ) -> VortexResult<Option<ArrayData>> {
        // We only support comparing a constant array to another constant array.
        // For all other encodings, we assume the constant is on the RHS.
        if let Some(const_scalar) = rhs.as_constant() {
            let lhs_scalar = lhs.scalar();
            let scalar = scalar_cmp(&lhs_scalar, &const_scalar, operator);
            return Ok(Some(ConstantArray::new(scalar, lhs.len()).into_array()));
        }

        Ok(None)
    }
}
