// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::Mask;
use vortex_mask::MaskValuesRef;

use super::super::RowFnExecutionArgs;
use super::super::args::BorrowedRowFnArgs;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::builtins::ArrayBuiltins;

impl RowFnExecutionArgs {
    /// Resolve validity, then execute valid rows over the original inputs.
    ///
    /// # Panics
    ///
    /// Panics if the concrete row signature cannot use direct valid-row execution. Inputs must
    /// support null-tolerant decoding, and output sinks must initialize skipped rows.
    pub(super) fn execute_valid_only(
        &self,
        kernel: impl Fn(BorrowedRowFnArgs<'_>, &mut ExecutionCtx) -> VortexResult<ArrayRef>,
        try_valid_rows: impl FnOnce(
            BorrowedRowFnArgs<'_>,
            MaskValuesRef,
            &mut ExecutionCtx,
        ) -> VortexResult<Option<ArrayRef>>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let validity = self.validity.clone().execute_mask(self.row_count, ctx)?;

        let valid_rows = match validity {
            // An empty mask is both all-valid and all-null. Preserve the all-valid behavior.
            Mask::AllTrue(_) | Mask::AllFalse(0) => {
                let values = kernel(self.execution_args(&self.inputs, self.row_count), ctx)?;
                let values = self.validate_kernel_output(values, self.row_count, ctx)?;

                return self.finalize_output(values, self.row_count);
            }
            Mask::AllFalse(_) => return Ok(self.all_null()),
            Mask::Values(valid_rows) => valid_rows,
        };

        if let Some(result) = self.try_execute_valid_rows(try_valid_rows, valid_rows, ctx)? {
            return Ok(result);
        }

        vortex_panic!(
            "valid-only execution requires direct valid-row support; {} selected an unsupported signature",
            self.id,
        )
    }

    /// Try execution against the original inputs, then mask a returned full-length result.
    pub(super) fn try_execute_valid_rows(
        &self,
        try_valid_rows: impl FnOnce(
            BorrowedRowFnArgs<'_>,
            MaskValuesRef,
            &mut ExecutionCtx,
        ) -> VortexResult<Option<ArrayRef>>,
        valid: MaskValuesRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(values) = try_valid_rows(
            self.execution_args(&self.inputs, self.row_count),
            MaskValuesRef::clone(&valid),
            ctx,
        )?
        else {
            return Ok(None);
        };
        let values = self.validate_kernel_output(values, valid.len(), ctx)?;

        let mask = valid.as_ref().into_array();
        self.finalize_output(values.mask(mask)?, valid.len())
            .map(Some)
    }
}
