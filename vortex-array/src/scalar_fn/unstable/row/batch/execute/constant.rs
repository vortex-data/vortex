// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::super::RowFnExecutionArgs;
use super::super::args::BorrowedRowFnArgs;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::validity::Validity;

impl RowFnExecutionArgs<'_> {
    /// Execute all-constant inputs by evaluating one row and broadcasting the validated result.
    pub(super) fn execute_all_constant(
        &self,
        kernel: impl FnOnce(
            BorrowedRowFnArgs<'_>,
            &mut Option<Validity>,
            &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let result = self.validate_kernel_output(
            kernel(self.execution_args(&self.batch.inputs, 1), &mut None, ctx)?,
            1,
            ctx,
        )?;
        let result = self.finalize_output(result, 1)?;
        let scalar = result.execute_scalar(0, ctx)?;

        Ok(ConstantArray::new(scalar, self.batch.row_count).into_array())
    }
}
