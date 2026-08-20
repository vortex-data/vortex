// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::super::RowFnExecutionArgs;
use super::super::args::BorrowedRowFnArgs;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::builtins::ArrayBuiltins;
use crate::validity::Validity;

impl RowFnExecutionArgs {
    /// Run every stored payload, then attach the input validity without materializing its mask.
    pub(super) fn execute_dense(
        &self,
        kernel: impl Fn(BorrowedRowFnArgs<'_>, &mut ExecutionCtx) -> VortexResult<ArrayRef>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let values = kernel(self.execution_args(&self.inputs, self.row_count), ctx)?;
        let values = self.validate_kernel_output(values, self.row_count, ctx)?;

        match self.validity.clone() {
            Validity::NonNullable | Validity::AllValid => {
                self.finalize_output(values, self.row_count)
            }
            Validity::Array(valid) => self.finalize_output(values.mask(valid)?, self.row_count),
            // Handled by the guard in `RowFnExecutionArgs::execute`, before the kernel ran.
            Validity::AllInvalid => Ok(self.all_null()),
        }
    }
}
