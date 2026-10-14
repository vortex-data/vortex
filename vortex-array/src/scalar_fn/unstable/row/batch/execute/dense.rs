// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_mask::MaskValuesRef;

use super::super::RowFnExecutionArgs;
use super::super::args::BorrowedRowFnArgs;
use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::masked::mask_validity_canonical;
use crate::builtins::ArrayBuiltins;
use crate::canonical::AnyCanonical;
use crate::scalar_fn::unstable::row::execute::DenseAttempt;
use crate::validity::Validity;

impl RowFnExecutionArgs<'_> {
    /// Run every stored payload, then attach the input validity without materializing its mask.
    pub(super) fn execute_dense(
        &self,
        kernel: impl FnOnce(
            BorrowedRowFnArgs<'_>,
            &mut Option<Validity>,
            &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        // Offer a partial validity to the kernel. A collector that takes it builds the output with
        // that validity attached, so the output needs neither a second construction nor an
        // all-valid check.
        let mut validity =
            matches!(self.batch.validity, Validity::Array(_)).then(|| self.batch.validity.clone());
        let values = kernel(
            self.execution_args(&self.batch.inputs, self.batch.row_count),
            &mut validity,
            ctx,
        )?;
        if validity.is_none() && matches!(self.batch.validity, Validity::Array(_)) {
            return self.finalize_output(values, self.batch.row_count);
        }

        self.finalize_dense_output(values, ctx)
    }

    /// Run all stored payloads, retrying valid rows only when reduced failure evidence is rejected.
    ///
    /// Validity stays lazy on success because masking makes null-row values unobservable. Rejected
    /// evidence loses which row failed, so validity decides whether to return, suppress, or
    /// recompute the error.
    pub(super) fn execute_dense_with_retry(
        &self,
        execute_dense_attempt: impl FnOnce(
            BorrowedRowFnArgs<'_>,
            &mut ExecutionCtx,
        ) -> VortexResult<DenseAttempt>,
        try_valid_rows: impl FnOnce(
            BorrowedRowFnArgs<'_>,
            MaskValuesRef,
            &mut ExecutionCtx,
        ) -> VortexResult<Option<ArrayRef>>,
        execute_filtered_rows: impl FnOnce(
            BorrowedRowFnArgs<'_>,
            MaskValuesRef,
            &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let attempt = execute_dense_attempt(
            self.execution_args(&self.batch.inputs, self.batch.row_count),
            ctx,
        )?;

        match attempt {
            DenseAttempt::Values(values) => self.finalize_dense_output(values, ctx),
            DenseAttempt::DeferredError(error) => {
                self.resolve_deferred_error(error, try_valid_rows, execute_filtered_rows, ctx)
            }
        }
    }

    fn resolve_deferred_error(
        &self,
        deferred_error: VortexError,
        try_valid_rows: impl FnOnce(
            BorrowedRowFnArgs<'_>,
            MaskValuesRef,
            &mut ExecutionCtx,
        ) -> VortexResult<Option<ArrayRef>>,
        execute_filtered_rows: impl FnOnce(
            BorrowedRowFnArgs<'_>,
            MaskValuesRef,
            &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let valid_rows = match self
            .batch
            .validity
            .execute_mask(self.batch.row_count, ctx)?
        {
            // An array-backed validity can materialize to all valid even though the cheap checks
            // in `RowFnExecutionArgs::try_execute_without_retry` could not prove that. The
            // deferred error therefore came from an observable row and remains terminal. An empty
            // mask is both all-valid and all-null, so preserve the all-valid behavior.
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

        self.execute_filtered(execute_filtered_rows, &valid_rows, ctx)
    }

    fn finalize_dense_output(
        &self,
        values: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let values = self.validate_kernel_output(values, self.batch.row_count, ctx)?;

        match self.batch.validity.clone() {
            Validity::NonNullable | Validity::AllValid => {
                self.finalize_output(values, self.batch.row_count)
            }
            // A canonical kernel output takes the validity directly. Masking it lazily would add
            // a scalar-function node that the caller must execute again just to attach it.
            Validity::Array(valid) => {
                let values = match values.as_opt::<AnyCanonical>() {
                    Some(canonical) => mask_validity_canonical(
                        Canonical::from(canonical),
                        Validity::Array(valid),
                        ctx,
                    )?
                    .into_array(),
                    None => values.mask(valid)?,
                };
                self.finalize_output(values, self.batch.row_count)
            }
            Validity::AllInvalid => {
                unreachable!("all-invalid validity is handled before dense row execution")
            }
        }
    }
}
