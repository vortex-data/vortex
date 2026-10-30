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
        let offered = matches!(self.batch.probe.validity, Validity::Array(_));
        let mut validity = offered.then(|| self.batch.probe.validity.clone());
        let values = kernel(
            self.execution_args(&self.batch.inputs, self.batch.probe.row_count),
            &mut validity,
            ctx,
        )?;

        // A taken validity proves nothing alone. Row kernels build non-nullable storage, so only a
        // nullable output carries the validity. Mask every other output, including one from a
        // collector that took the validity without attaching it.
        if offered && validity.is_none() && values.dtype().is_nullable() {
            return self.finalize_output(values, self.batch.probe.row_count);
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
            self.execution_args(&self.batch.inputs, self.batch.probe.row_count),
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
            .probe
            .validity
            .execute_mask(self.batch.probe.row_count, ctx)?
        {
            // An array-backed validity can materialize to all valid even though the cheap checks
            // in `BatchInputs::try_execute_without_retry` could not prove that. The deferred error
            // therefore came from an observable row and remains terminal. An empty mask is both
            // all-valid and all-null, so preserve the all-valid behavior.
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
        let values = self.validate_kernel_output(values, self.batch.probe.row_count, ctx)?;

        match self.batch.probe.validity.clone() {
            Validity::NonNullable | Validity::AllValid => {
                self.finalize_output(values, self.batch.probe.row_count)
            }
            Validity::Array(valid) => {
                // `values.mask(valid)` builds and optimizes a ScalarFnArray to dispatch through
                // parent-reduce rules. Benchmarks motivated bypassing that setup here, keeping the
                // exception internal instead of adding canonical-specific dispatch to the generic
                // builtin. Remove this bypass if `.mask()` gains equivalent direct dispatch.
                // https://github.com/vortex-data/vortex/pull/10742
                let values = match values.as_opt::<AnyCanonical>() {
                    Some(canonical) => mask_validity_canonical(
                        Canonical::from(canonical),
                        Validity::Array(valid),
                        ctx,
                    )?
                    .into_array(),
                    None => values.mask(valid)?,
                };
                self.finalize_output(values, self.batch.probe.row_count)
            }
            Validity::AllInvalid => {
                unreachable!("all-invalid validity is handled before dense row execution")
            }
        }
    }
}
