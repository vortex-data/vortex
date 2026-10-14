// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Selects a batch execution strategy.
//!
//! [`RowFnExecutionArgs::try_execute_without_retry`] handles universal fast paths and dense
//! execution, so the visit that planned a batch can finish it without dispatching again.
//! [`RowFnExecutionArgs::execute_partially_valid`] then delegates the remaining partially valid
//! batches to dense retry or valid-only execution.

use vortex_error::VortexResult;
use vortex_mask::MaskValuesRef;

use super::RowFnExecutionArgs;
use super::RowPolicy;
use super::args::BorrowedRowFnArgs;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::scalar_fn::unstable::row::execute::DenseAttempt;
use crate::validity::Validity;

mod constant;
mod dense;
mod filtered;
mod valid_only;

mod output;
pub(crate) use output::finalize_kernel_output;

impl RowFnExecutionArgs<'_> {
    /// Execute every batch that `kernel` alone can finish, or return `None` for a partially valid
    /// batch whose plan needs a dense retry or valid-row execution.
    ///
    /// `kernel` reports a rejected deferred failure as a terminal error. That is correct for each
    /// batch this method executes: no null row can be the cause of the failure.
    pub(crate) fn try_execute_without_retry(
        &self,
        kernel: impl FnOnce(
            BorrowedRowFnArgs<'_>,
            &mut Option<Validity>,
            &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Strictness: an all-null batch has no observable row work. Keep the literal-constant
        // check explicit alongside the conjoined validity invariant.
        if matches!(self.batch.validity, Validity::AllInvalid) || self.batch.has_null_const {
            return Ok(Some(self.all_null()));
        }

        // All inputs are constant, and their conjoined validity proves that every row is non-null.
        // The constant check sees through extension and masked wrappers, just like argument
        // decoding.
        if self.batch.row_count > 0
            && self.batch.validity.definitely_no_nulls()
            && self.batch.all_batch_const()
        {
            return self.execute_all_constant(kernel, ctx).map(Some);
        }

        // Do not resolve array-backed validity for the uncommon all-valid or all-null cases here.
        // That can execute and scan the full mask; each policy resolves it only when necessary.
        if self.batch.validity.definitely_no_nulls() || self.plan.policy() == RowPolicy::Dense {
            return self.execute_dense(kernel, ctx).map(Some);
        }

        Ok(None)
    }

    /// Execute a partially valid batch that
    /// [`try_execute_without_retry`](Self::try_execute_without_retry) declined.
    ///
    /// For a partially valid batch, `try_valid_rows` can avoid filtering. `Ok(None)` filters the
    /// inputs to the valid rows and `execute_filtered_rows` runs the kernel over them while
    /// writing into the original row domain. Every result is checked against the planned shape
    /// and dtype.
    pub(crate) fn execute_partially_valid(
        &self,
        kernel: impl FnOnce(
            BorrowedRowFnArgs<'_>,
            &mut Option<Validity>,
            &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef>,
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
        match self.plan.policy() {
            RowPolicy::Dense => self.execute_dense(kernel, ctx),
            RowPolicy::DenseWithRetry => self.execute_dense_with_retry(
                execute_dense_attempt,
                try_valid_rows,
                execute_filtered_rows,
                ctx,
            ),
            RowPolicy::ValidOnly => {
                self.execute_valid_only(kernel, try_valid_rows, execute_filtered_rows, ctx)
            }
        }
    }
}
