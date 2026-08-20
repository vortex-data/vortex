// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
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
    /// Resolve validity and try direct valid-row execution before filter-and-scatter.
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
        let original_validity = match self.resolve_validity(&kernel, ctx)? {
            ResolvedValidity::Output(output) => return Ok(output),
            ResolvedValidity::PartiallyValid(original_validity) => original_validity,
        };

        let direct_output = self.try_execute_valid_rows(try_valid_rows, &original_validity, ctx)?;

        if let Some(output) = direct_output {
            return Ok(output);
        }

        self.filter_and_scatter(kernel, &original_validity, ctx)
    }

    /// Materialize validity and handle all-valid or all-null batches.
    fn resolve_validity(
        &self,
        kernel: &impl Fn(BorrowedRowFnArgs<'_>, &mut ExecutionCtx) -> VortexResult<ArrayRef>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ResolvedValidity> {
        let validity = self.validity.clone().execute_mask(self.row_count, ctx)?;

        // An array-backed validity can materialize to all valid even though the cheap checks in
        // `RowFnExecutionArgs::execute` could not prove that. Run the full-row kernel in that
        // case. Check all-true before all-false because an empty mask is both.
        if validity.all_true() {
            let values = kernel(self.execution_args(&self.inputs, self.row_count), ctx)?;
            let values = self.validate_kernel_output(values, self.row_count, ctx)?;
            let values = self.finalize_output(values, self.row_count)?;

            return Ok(ResolvedValidity::Output(values));
        }

        if validity.all_false() {
            return Ok(ResolvedValidity::Output(self.all_null()));
        }

        Ok(ResolvedValidity::PartiallyValid(validity))
    }

    /// Try execution against the original inputs, then mask a returned full-length result.
    fn try_execute_valid_rows(
        &self,
        try_valid_rows: impl FnOnce(
            BorrowedRowFnArgs<'_>,
            &Mask,
            &mut ExecutionCtx,
        ) -> VortexResult<Option<ArrayRef>>,
        original_validity: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let row_count = original_validity.len();
        let args = self.execution_args(&self.inputs, row_count);

        let Some(values) = try_valid_rows(args, original_validity, ctx)? else {
            return Ok(None);
        };

        let values = self.validate_kernel_output(values, row_count, ctx)?;

        let validity_array =
            BoolArray::new(original_validity.to_bit_buffer(), Validity::NonNullable).into_array();
        let masked = values.mask(validity_array)?;

        self.finalize_output(masked, row_count).map(Some)
    }
}
