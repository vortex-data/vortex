use vortex_error::{VortexResult, vortex_err};
use vortex_scalar::NumericOperator;

use crate::arrays::{ConstantArray, ConstantEncoding};
use crate::compute::{NumericKernel, NumericKernelAdapter};
use crate::{Array, ArrayRef, register_kernel};

impl NumericKernel for ConstantEncoding {
    fn numeric(
        &self,
        array: &ConstantArray,
        rhs: &dyn Array,
        op: NumericOperator,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(rhs) = rhs.as_constant() else {
            return Ok(None);
        };

        Ok(Some(
            ConstantArray::new(
                array
                    .scalar()
                    .as_primitive()
                    .checked_binary_numeric(&rhs.as_primitive(), op)
                    .ok_or_else(|| vortex_err!("numeric overflow"))?,
                array.len(),
            )
            .into_array(),
        ))
    }
}

register_kernel!(NumericKernelAdapter(ConstantEncoding).lift());
