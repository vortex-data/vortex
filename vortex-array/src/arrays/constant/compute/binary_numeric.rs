use vortex_error::{vortex_err, VortexResult};
use vortex_scalar::NumericOperator;

use crate::arrays::{ConstantArray, ConstantVTable};
use crate::compute::{NumericKernel, NumericKernelAdapter};
use crate::{register_kernel, Array, ArrayRef, IntoArray};

impl NumericKernel for ConstantVTable {
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

register_kernel!(NumericKernelAdapter(ConstantVTable).lift());
