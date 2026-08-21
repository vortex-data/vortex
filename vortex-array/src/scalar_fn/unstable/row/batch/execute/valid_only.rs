// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use super::super::RowFnExecutionArgs;
use super::super::args::BorrowedRowFnArgs;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::builtins::ArrayBuiltins;
use crate::validity::Validity;

/// The result of resolving batch validity.
enum ResolvedValidity {
    /// The output for an all-valid or all-null batch.
    Output(ArrayRef),

    /// A mask with both valid and invalid rows.
    PartiallyValid(Mask),
}

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
            &Mask,
            &mut ExecutionCtx,
        ) -> VortexResult<Option<ArrayRef>>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let valid = match self.resolve_validity(&kernel, ctx)? {
            ResolvedValidity::Output(output) => return Ok(output),
            ResolvedValidity::PartiallyValid(valid) => valid,
        };

        if let Some(result) = self.try_execute_valid_rows(try_valid_rows, &valid, ctx)? {
            return Ok(result);
        }

        vortex_panic!(
            "valid-only execution requires direct valid-row support; {} selected an unsupported signature",
            self.id,
        )
    }

    /// Materialize validity and handle all-valid or all-null batches.
    fn resolve_validity(
        &self,
        kernel: &impl Fn(BorrowedRowFnArgs<'_>, &mut ExecutionCtx) -> VortexResult<ArrayRef>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ResolvedValidity> {
        let valid = self.validity.clone().execute_mask(self.row_count, ctx)?;

        // An array-backed validity can materialize to all valid even though the cheap checks in
        // `RowFnExecutionArgs::execute` could not prove that. Run the full-row kernel in that
        // case. Check all-true before all-false because an empty mask is both.
        if valid.all_true() {
            let values = kernel(self.execution_args(&self.inputs, self.row_count), ctx)?;
            let values = self.validate_kernel_output(values, self.row_count, ctx)?;
            let values = self.finalize_output(values, self.row_count)?;

            return Ok(ResolvedValidity::Output(values));
        }

        if valid.all_false() {
            return Ok(ResolvedValidity::Output(self.all_null()));
        }

        Ok(ResolvedValidity::PartiallyValid(valid))
    }

    /// Try execution against the original inputs, then mask a returned full-length result.
    pub(super) fn try_execute_valid_rows(
        &self,
        try_valid_rows: impl FnOnce(
            BorrowedRowFnArgs<'_>,
            &Mask,
            &mut ExecutionCtx,
        ) -> VortexResult<Option<ArrayRef>>,
        valid: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(values) = try_valid_rows(
            self.execution_args(&self.inputs, self.row_count),
            valid,
            ctx,
        )?
        else {
            return Ok(None);
        };
        let values = self.validate_kernel_output(values, valid.len(), ctx)?;

        let mask = BoolArray::new(valid.to_bit_buffer(), Validity::NonNullable).into_array();
        self.finalize_output(values.mask(mask)?, valid.len())
            .map(Some)
    }
}
