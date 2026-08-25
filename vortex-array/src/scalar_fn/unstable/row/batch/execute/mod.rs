// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Selects a batch execution strategy.
//!
//! [`RowFnExecutionArgs::execute`] handles universal fast paths, then delegates to dense or
//! valid-only execution.

use vortex_error::VortexResult;
use vortex_mask::MaskValuesRef;

use super::RowFnExecutionArgs;
use super::RowPolicy;
use super::args::BorrowedRowFnArgs;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::Constant;
use crate::scalar_fn::unstable::row::execute::DenseAttempt;
use crate::scalar_fn::unstable::row::types::batch_const;
use crate::validity::Validity;

mod constant;
mod dense;
mod valid_only;

mod output;
pub(crate) use output::finalize_kernel_output;

impl RowFnExecutionArgs {
    /// Apply constant folding and null handling around `kernel`.
    ///
    /// For a partially valid batch, `try_valid_rows` executes only valid rows over the original
    /// inputs. Every result is checked against the planned shape and dtype.
    pub(crate) fn execute(
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
        // Strictness: an all-null batch has no observable row work. Keep the literal-constant
        // check explicit alongside the conjoined validity invariant.
        if matches!(self.validity, Validity::AllInvalid)
            || self.inputs.iter().any(|input| {
                input
                    .as_opt::<Constant>()
                    .is_some_and(|constant| constant.scalar().is_null())
            })
        {
            return Ok(self.all_null());
        }

        // All inputs are constant, and their conjoined validity proves that every row is non-null.
        // The constant check sees through extension and masked wrappers, just like argument
        // decoding.
        if self.row_count > 0
            && self.validity.definitely_no_nulls()
            && self.inputs.iter().all(|input| batch_const(input).is_some())
        {
            return self.execute_all_constant(kernel, ctx);
        }

        // Do not resolve array-backed validity for the uncommon all-valid or all-null cases here.
        // That can execute and scan the full mask; each policy resolves it only when necessary.
        if self.validity.definitely_no_nulls() {
            return self.execute_dense(kernel, ctx);
        }

        match self.policy {
            RowPolicy::Dense => self.execute_dense(kernel, ctx),
            RowPolicy::DenseWithRetry => {
                self.execute_dense_with_retry(execute_dense_attempt, try_valid_rows, ctx)
            }
            RowPolicy::ValidOnly => self.execute_valid_only(kernel, try_valid_rows, ctx),
        }
    }
}
