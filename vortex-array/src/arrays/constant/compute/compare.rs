use vortex_error::VortexResult;

use crate::arrays::{ConstantArray, ConstantVTable};
use crate::compute::{scalar_cmp, CompareKernel, CompareKernelAdapter, Operator};
use crate::{register_kernel, Array, ArrayRef, IntoArray};

impl CompareKernel for ConstantVTable {
    fn compare(
        &self,
        lhs: &ConstantArray,
        rhs: &dyn Array,
        operator: Operator,
    ) -> VortexResult<Option<ArrayRef>> {
        // We only support comparing a constant array to another constant array.
        // For all other encodings, we assume the constant is on the RHS.
        if let Some(const_scalar) = rhs.as_constant() {
            let lhs_scalar = lhs.scalar();
            let scalar = scalar_cmp(lhs_scalar, &const_scalar, operator);
            return Ok(Some(ConstantArray::new(scalar, lhs.len()).into_array()));
        }

        Ok(None)
    }
}

register_kernel!(CompareKernelAdapter(ConstantVTable).lift());
