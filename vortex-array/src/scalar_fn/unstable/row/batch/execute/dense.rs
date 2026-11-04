// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution over every stored payload, including payloads in null rows.
//!
//! When the input types and kernel tolerate null-row payloads, a contiguous row loop avoids
//! materializing validity, filtering the inputs, or testing validity at each row. Batch validity
//! makes the results computed from null payloads unobservable.
//!
//! Most kernels build all-valid storage that this module masks afterward. The Boolean collector
//! already controls construction of its final array, so it can attach batch validity in that
//! constructor. Passing validity into the kernel avoids constructing an all-valid array only to
//! replace it with a masked array, and keeps array-backed validity lazy.

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
    ///
    /// The caller must establish that every row is valid or that the plan permits evaluating
    /// null-row payloads. The callback receives `Some(batch_validity)` only when that validity is
    /// array-backed, and must follow one of these paths:
    ///
    /// - Leave the option unchanged and return all-valid kernel output. This method validates the
    ///   output and attaches batch validity afterward.
    /// - Call `take()` and return a nullable array carrying that exact validity. This method then
    ///   checks the output's length and dtype without requiring every row to be valid or masking
    ///   the array again.
    ///
    /// An initial `None` must remain `None`, and its kernel output must be all-valid. The callback
    /// **must not** insert or replace validity. Taking validity without attaching it can produce
    /// incorrect nulls: the executor trusts the callback and does not compare the returned array's
    /// validity with the original batch validity.
    pub(super) fn execute_dense(
        &self,
        kernel: impl FnOnce(
            BorrowedRowFnArgs<'_>,
            &mut Option<Validity>,
            &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        // Only array-backed validity carries a per-row mask. The Boolean constructor can attach
        // it without executing that mask. Known all-valid inputs need no such handoff.
        let has_array_validity = matches!(self.batch.probe.validity, Validity::Array(_));
        let mut validity = has_array_validity.then(|| self.batch.probe.validity.clone());
        let values = kernel(
            self.execution_args(&self.batch.inputs, self.batch.probe.row_count),
            &mut validity,
            ctx,
        )?;

        // If `kernel` took `Some(validity)` and built a nullable result, its contract requires that
        // it attached that validity. The all-valid kernel check would now execute the validity and
        // reject propagated input nulls. Only length, dtype, and logical nullability need checking.
        if has_array_validity && validity.is_none() && values.dtype().is_nullable() {
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
            // The cheap checks in `BatchInputs::try_execute_without_retry` cannot see that an
            // array-backed validity is all valid. If it is, a valid row caused the deferred error,
            // so the error is final. An empty mask counts as all valid.
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
                // `values.mask(valid)` builds and optimizes a scalar-function expression before
                // reaching canonical masking. Attach validity directly when the output is already
                // canonical to avoid that setup. Keeping this shortcut here leaves the generic
                // builtin independent of canonical encodings.
                // TODO(connor): Replace this shortcut with `values.mask(valid)` once the builtin
                // can dispatch directly to canonical validity replacement without building and
                // optimizing a scalar-function expression.
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
