// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Selects a batch execution strategy.
//!
//! [`Batch::execute`] handles universal fast paths, then delegates to dense or valid-only
//! execution.

use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::Batch;
use super::RowPolicy;
use super::args::BorrowedExecutionArgs;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::Constant;
use crate::scalar_fn::unstable::row::types::batch_const;
use crate::validity::Validity;

mod constant;
mod dense;
mod valid_only;

mod output;
#[cfg(test)]
pub(crate) use output::finalize_kernel_output;

impl Batch {
    /// Apply constant folding and null handling around `kernel`.
    ///
    /// When the mask contains valid and invalid rows, `try_valid_rows` executes only valid rows
    /// over the original inputs. Every result is checked against the planned shape and dtype.
    pub(crate) fn execute(
        &self,
        kernel: impl Fn(BorrowedExecutionArgs<'_>, &mut ExecutionCtx) -> VortexResult<ArrayRef>,
        try_valid_rows: impl FnOnce(
            BorrowedExecutionArgs<'_>,
            &Mask,
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

        // All inputs constant, and their conjoined validity proves every row non-null. This sees
        // through extension and masked wrappers just like argument decoding does.
        if self.row_count > 0
            && self.validity.definitely_no_nulls()
            && self.inputs.iter().all(|input| batch_const(input).is_some())
        {
            return self.execute_all_constant(kernel, ctx);
        }

        // A known all-valid batch does not need to materialize validity, even when its row policy
        // only permits valid rows.
        if self.validity.definitely_no_nulls() {
            return self.execute_dense(kernel, ctx);
        }

        match self.policy {
            RowPolicy::Dense => self.execute_dense(kernel, ctx),
            RowPolicy::ValidOnly => self.execute_valid_only(kernel, try_valid_rows, ctx),
        }
    }
}
