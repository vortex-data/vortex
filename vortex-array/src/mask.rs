// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::BitAnd;

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use crate::Array;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::executor::CanonicalOutput;
use crate::executor::VectorExecutor;

/// Executor for exporting a Vortex [`Mask`] from an [`ArrayRef`].
pub trait MaskExecutor {
    /// Execute the array to produce a mask.
    fn execute_mask(&self, ctx: &mut ExecutionCtx) -> VortexResult<Mask>;
}

impl MaskExecutor for ArrayRef {
    fn execute_mask(&self, ctx: &mut ExecutionCtx) -> VortexResult<Mask> {
        if !matches!(self.dtype(), DType::Bool(_)) {
            vortex_bail!("Mask array must have boolean dtype, not {}", self.dtype());
        }

        Ok(match self.execute_output(ctx)? {
            CanonicalOutput::Constant(c) => {
                Mask::new(self.len(), c.scalar().as_bool().value().unwrap_or(false))
            }
            CanonicalOutput::Array(a) => {
                let (bits, mask) = a.to_vector(ctx)?.into_bool().into_parts();
                // To handle nullable boolean arrays, we treat nulls as false in the mask.
                // TODO(ngates): is this correct? Feels like we should just force the caller to
                //  pass non-nullable boolean arrays.
                mask.bitand(&Mask::from(bits))
            }
        })
    }
}
