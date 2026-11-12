// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Selects a batch execution strategy.
//!
//! [`BatchInputs::try_execute_without_retry`] finishes every batch that at most one kernel call can
//! finish, inside the planning visit. The rest come back as a [`PartiallyValidBatch`] for a dense
//! retry or valid-row execution. Finishing inside the visit reuses its row closure, avoiding
//! another dispatch to recover the same kernel.

use vortex_error::VortexResult;
use vortex_mask::MaskValuesRef;

use super::BatchInputs;
use super::BatchPlan;
use super::PartiallyValidBatch;
use super::RowPolicy;
use super::args::BorrowedRowFnArgs;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::scalar_fn::unstable::row::execute::DenseAttempt;
use crate::validity::Validity;

mod constant;
mod dense;
mod filtered;
mod valid_only;

mod output;
pub(crate) use output::finalize_kernel_output;

impl<'a> BatchInputs<'a> {
    /// Execute this batch under `plan` when at most one call to `kernel` can finish it. Otherwise
    /// return it as a [`PartiallyValidBatch`] that needs a dense retry or valid-row execution.
    ///
    /// The callback's optional validity argument follows the [dense execution contract]. It lets a
    /// kernel attach batch validity during output construction, avoiding a second array construction
    /// for masking. Other kernels leave it unchanged and return all-valid output. All-constant
    /// execution passes `None` because it evaluates one valid row instead of constructing the full
    /// batch.
    ///
    /// A deferred kernel does not report which row failed, and `kernel` returns its failure as an
    /// error. That is correct here because this method runs a deferred kernel only when no row is
    /// null, so the failing row is valid.
    ///
    /// [dense execution contract]: super::RowFnExecutionArgs::execute_dense
    pub(crate) fn try_execute_without_retry(
        self,
        arg_dtypes: &[DType],
        plan: BatchPlan,
        kernel: impl FnOnce(
            BorrowedRowFnArgs<'_>,
            &mut Option<Validity>,
            &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Result<ArrayRef, PartiallyValidBatch<'a>>> {
        // Every output row is null, so skip the kernel. Checking `has_null_const` as well keeps
        // this from relying on `BatchProbe::new` to mark a null literal as `AllInvalid`.
        if matches!(self.probe.validity, Validity::AllInvalid) || self.probe.has_null_const {
            return Ok(Ok(self.with_plan(arg_dtypes, plan).all_null()));
        }

        // Every input is constant and no row is null, so evaluate one row instead of decoding and
        // allocating an output for the full batch. An empty batch has no row to evaluate.
        // `all_batch_const` sees through extension and masked wrappers, as argument decoding does.
        if self.probe.row_count > 0
            && self.probe.validity.definitely_no_nulls()
            && self.all_batch_const()
        {
            return self
                .with_plan(arg_dtypes, plan)
                .execute_all_constant(kernel, ctx)
                .map(Ok);
        }

        // A contiguous loop needs no validity scan when every row is known valid or the policy
        // permits evaluating null-row payloads. Other policies must resolve validity first so a
        // null row cannot cause a decode or kernel error that masking would fail to hide.
        if self.probe.validity.definitely_no_nulls() || plan.policy() == RowPolicy::Dense {
            return self
                .with_plan(arg_dtypes, plan)
                .execute_dense(kernel, ctx)
                .map(Ok);
        }

        // `PartiallyValidBatch::execute` attaches the plan, so attaching it here would compute the
        // result dtype twice.
        Ok(Err(PartiallyValidBatch { batch: self, plan }))
    }
}

impl PartiallyValidBatch<'_> {
    /// Execute the batch with its planned strategy: a dense retry or valid-row execution.
    ///
    /// `try_valid_rows` can run the valid rows without filtering. If it returns `Ok(None)`, the
    /// inputs are filtered to the valid rows, and `execute_filtered_rows` writes each result back
    /// at its original row. Every result is checked against the planned length and dtype.
    pub(crate) fn execute(
        self,
        arg_dtypes: &[DType],
        kernel: impl FnOnce(BorrowedRowFnArgs<'_>, &mut ExecutionCtx) -> VortexResult<ArrayRef>,
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
        let Self { batch, plan } = self;
        let args = batch.with_plan(arg_dtypes, plan);

        match args.plan.policy() {
            RowPolicy::Dense => {
                unreachable!("BatchInputs::try_execute_without_retry executes every dense batch")
            }
            RowPolicy::DenseWithRetry => args.execute_dense_with_retry(
                execute_dense_attempt,
                try_valid_rows,
                execute_filtered_rows,
                ctx,
            ),
            RowPolicy::ValidOnly => {
                args.execute_valid_only(kernel, try_valid_rows, execute_filtered_rows, ctx)
            }
        }
    }
}
