// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_mask::MaskValuesRef;

use super::super::RowFnExecutionArgs;
use super::super::args::BorrowedRowFnArgs;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::builtins::ArrayBuiltins;
use crate::scalar_fn::unstable::row::execute::DenseAttempt;
use crate::validity::Validity;

impl RowFnExecutionArgs {
    /// Run every stored payload, then attach the input validity without materializing its mask.
    pub(super) fn execute_dense(
        &self,
        kernel: impl Fn(BorrowedRowFnArgs<'_>, &mut ExecutionCtx) -> VortexResult<ArrayRef>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let values = kernel(self.execution_args(&self.inputs, self.row_count), ctx)?;

        self.finalize_dense_output(values, ctx)
    }

    /// Run all stored payloads, retrying valid rows only when reduced failure evidence is rejected.
    ///
    /// Validity stays lazy on success because masking makes null-row values unobservable. Rejected
    /// evidence loses which row failed, so validity decides whether to return, suppress, or
    /// recompute the error.
    pub(super) fn execute_dense_with_retry(
        &self,
        kernel: impl Fn(BorrowedRowFnArgs<'_>, &mut ExecutionCtx) -> VortexResult<ArrayRef>,
        execute_dense_attempt: impl FnOnce(
            BorrowedRowFnArgs<'_>,
            &mut ExecutionCtx,
        ) -> VortexResult<DenseAttempt>,
        try_valid_rows: impl FnOnce(
            BorrowedRowFnArgs<'_>,
            MaskValuesRef,
            &mut ExecutionCtx,
        ) -> VortexResult<Option<ArrayRef>>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let attempt =
            execute_dense_attempt(self.execution_args(&self.inputs, self.row_count), ctx)?;

        match attempt {
            DenseAttempt::Values(values) => self.finalize_dense_output(values, ctx),
            DenseAttempt::DeferredError(error) => {
                self.resolve_deferred_error(error, kernel, try_valid_rows, ctx)
            }
        }
    }

    fn resolve_deferred_error(
        &self,
        deferred_error: VortexError,
        kernel: impl Fn(BorrowedRowFnArgs<'_>, &mut ExecutionCtx) -> VortexResult<ArrayRef>,
        try_valid_rows: impl FnOnce(
            BorrowedRowFnArgs<'_>,
            MaskValuesRef,
            &mut ExecutionCtx,
        ) -> VortexResult<Option<ArrayRef>>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let valid_rows = match self.validity.execute_mask(self.row_count, ctx)? {
            // An array-backed validity can materialize to all valid even though the cheap checks
            // in `RowFnExecutionArgs::execute` could not prove that. The deferred error therefore
            // came from an observable row and remains terminal. An empty mask is both all-valid
            // and all-null, so preserve the all-valid behavior.
            Mask::AllTrue(_) | Mask::AllFalse(0) => return Err(deferred_error),
            Mask::AllFalse(_) => return Ok(self.all_null()),
            Mask::Values(valid_rows) => valid_rows,
        };

        // Reduced evidence does not identify which rows failed. Discard the dense error before
        // retrying only observable rows.
        drop(deferred_error);

        if let Some(result) = self.try_execute_valid_rows(try_valid_rows, &valid_rows, ctx)? {
            return Ok(result);
        }

        self.filter_and_scatter(kernel, &valid_rows, ctx)
    }

    fn finalize_dense_output(
        &self,
        values: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let values = self.validate_kernel_output(values, self.row_count, ctx)?;

        match self.validity.clone() {
            Validity::NonNullable | Validity::AllValid => {
                self.finalize_output(values, self.row_count)
            }
            Validity::Array(valid) => self.finalize_output(values.mask(valid)?, self.row_count),
            Validity::AllInvalid => {
                unreachable!("all-invalid validity is handled before dense row execution")
            }
        }
    }
}
