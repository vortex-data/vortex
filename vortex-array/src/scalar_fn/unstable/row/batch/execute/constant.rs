// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::super::RowFnExecutionArgs;
use super::super::args::BorrowedRowFnArgs;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::scalar_fn::unstable::row::RowOutput;
use crate::validity::Validity;

impl RowFnExecutionArgs {
    /// Execute all-constant inputs by evaluating one row and broadcasting the validated result.
    pub(super) fn execute_all_constant(
        &self,
        kernel: impl Fn(BorrowedRowFnArgs<'_>, &mut ExecutionCtx) -> VortexResult<RowOutput>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let result = self.finish_kernel_output(
            kernel(self.execution_args(&self.inputs, 1), ctx)?,
            1,
            Validity::from(self.result_dtype.nullability()),
            ctx,
        )?;
        let result = self.finalize_output(result, 1)?;
        let scalar = result.execute_scalar(0, ctx)?;

        Ok(ConstantArray::new(scalar, self.row_count).into_array())
    }
}
