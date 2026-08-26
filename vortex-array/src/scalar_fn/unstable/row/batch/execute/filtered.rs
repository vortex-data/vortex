// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Filtered execution for row kernels that cannot operate on the original partially valid inputs.
//!
//! Direct skip-invalid execution retains the original input columns: it decodes every column,
//! initializes a full-length output, and visits only valid rows. Some input representations cannot
//! decode unspecified payloads behind nulls, so there is no safe row loop to enter over the
//! original columns.
//!
//! This path filters every input to the valid row domain before decoding it. The kernel then
//! iterates only the valid rows: it reads consecutive rows from the compact, all-valid inputs and
//! writes each result directly at its original row index into a full-length output. Skipped
//! positions hold placeholders that batch execution masks, exactly like direct skip-invalid
//! execution, so the output never needs a columnar scatter.
//!
//! Filtering nested or compressed inputs can materialize their selected representation, so batch
//! execution tries direct skip-invalid execution first and filters only when a required input
//! capability is unavailable. Calling the general scalar-extraction API once per valid row is not
//! equivalent because it repeats array execution and scalar construction inside the row loop.

use smallvec::SmallVec;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_mask::MaskValuesRef;

use super::super::RowFnExecutionArgs;
use super::super::args::BorrowedRowFnArgs;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::builtins::ArrayBuiltins;

impl RowFnExecutionArgs {
    /// Filter the batch to valid rows, then execute the kernel into the original row domain.
    pub(super) fn execute_filtered(
        &self,
        execute_filtered_rows: impl FnOnce(
            BorrowedRowFnArgs<'_>,
            MaskValuesRef,
            &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef>,
        valid: &MaskValuesRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let filtered_len = valid.true_count();
        let filter_mask = Mask::Values(MaskValuesRef::clone(valid));

        let filtered_inputs: SmallVec<[ArrayRef; 4]> = self
            .inputs
            .iter()
            .map(|input| input.filter(filter_mask.clone()))
            .collect::<VortexResult<_>>()?;

        let filtered_args = self.execution_args(&filtered_inputs, filtered_len);
        let values = execute_filtered_rows(filtered_args, MaskValuesRef::clone(valid), ctx)?;
        let values = self.validate_kernel_output(values, self.row_count, ctx)?;

        let mask = valid.as_ref().into_array();
        self.finalize_output(values.mask(mask)?, self.row_count)
    }
}
