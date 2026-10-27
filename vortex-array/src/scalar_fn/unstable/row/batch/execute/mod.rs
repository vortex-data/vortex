// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Selects a batch execution strategy.
//!
//! [`BatchInputs::try_execute_without_retry`] handles universal fast paths and dense execution, so
//! the visit that planned a batch can finish it without dispatching again. It hands every other
//! batch back as a [`PartiallyValidBatch`], which delegates to dense retry or valid-only execution.

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
    /// Execute this batch under `plan` when `kernel` alone can finish it. Otherwise return it as a
    /// [`PartiallyValidBatch`] that needs a dense retry or valid-row execution.
    ///
    /// `kernel` reports a rejected deferred failure as a terminal error. That is correct for each
    /// batch this method executes: no null row can be the cause of the failure.
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
        let args = self.with_plan(arg_dtypes, plan);

        // Strictness: an all-null batch has no observable row work. Keep the literal-constant
        // check explicit alongside the conjoined validity invariant.
        if matches!(self.probe.validity, Validity::AllInvalid) || self.probe.has_null_const {
            return Ok(Ok(args.all_null()));
        }

        // All inputs are constant, and their conjoined validity proves that every row is non-null.
        // The constant check sees through extension and masked wrappers, just like argument
        // decoding.
        if self.probe.row_count > 0
            && self.probe.validity.definitely_no_nulls()
            && self.all_batch_const()
        {
            return args.execute_all_constant(kernel, ctx).map(Ok);
        }

        // Do not resolve array-backed validity for the uncommon all-valid or all-null cases here.
        // That can execute and scan the full mask; each policy resolves it only when necessary.
        if self.probe.validity.definitely_no_nulls() || args.plan.policy() == RowPolicy::Dense {
            return args.execute_dense(kernel, ctx).map(Ok);
        }

        let plan = args.into_plan();

        Ok(Err(PartiallyValidBatch { batch: self, plan }))
    }
}

impl PartiallyValidBatch<'_> {
    /// Execute the partially valid batch with the dense retry or valid-row strategy of its plan.
    ///
    /// For a partially valid batch, `try_valid_rows` can avoid filtering. `Ok(None)` filters the
    /// inputs to the valid rows and `execute_filtered_rows` runs the kernel over them while
    /// writing into the original row domain. Every result is checked against the planned shape
    /// and dtype.
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
