// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_scalar::NumericOperator;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::compute::NumericKernel;
use crate::compute::NumericKernelAdapter;
use crate::register_kernel;

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
