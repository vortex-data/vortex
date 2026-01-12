// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_vector::Vector;

use crate::Array;
use crate::ArrayRef;
use crate::Executable;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantVTable;
use crate::executor::CanonicalOutput;

impl Executable for Mask {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        if !matches!(array.dtype(), DType::Bool(_)) {
            vortex_bail!("Mask array must have boolean dtype, not {}", array.dtype());
        }

        if let Some(constant) = array.as_opt::<ConstantVTable>() {
            let mask_value = constant.scalar().as_bool().value().unwrap_or(false);
            return Ok(Mask::new(array.len(), mask_value));
        }

        let array_len = array.len();
        Ok(match array.execute(ctx)? {
            CanonicalOutput::Constant(c) => {
                Mask::new(array_len, c.scalar().as_bool().value().unwrap_or(false))
            }
            CanonicalOutput::Array(a) => {
                let (bits, mask) = a
                    .into_array()
                    .execute::<Vector>(ctx)?
                    .into_bool()
                    .into_parts();
                // To handle nullable boolean arrays, we treat nulls as false in the mask.
                // TODO(ngates): is this correct? Feels like we should just force the caller to
                //  pass non-nullable boolean arrays.
                mask.bitand(&Mask::from(bits))
            }
        })
    }
}
