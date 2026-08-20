// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use smallvec::SmallVec;
use vortex_error::VortexResult;

use super::super::RowFnExecutionArgs;
use super::super::args::BorrowedRowFnArgs;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;

impl RowFnExecutionArgs {
    /// Execute all-constant inputs by evaluating one row and broadcasting the validated result.
    pub(super) fn execute_all_constant(
        &self,
        kernel: impl Fn(BorrowedRowFnArgs<'_>, &mut ExecutionCtx) -> VortexResult<ArrayRef>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let one_row: SmallVec<[ArrayRef; 4]> = self
            .inputs
            .iter()
            .map(|input| input.slice(0..1))
            .collect::<VortexResult<_>>()?;

        let result =
            self.validate_kernel_output(kernel(self.execution_args(&one_row, 1), ctx)?, 1, ctx)?;
        let result = self.finalize_output(result, 1)?;
        let scalar = result.execute_scalar(0, ctx)?;

        Ok(ConstantArray::new(scalar, self.row_count).into_array())
    }
}
