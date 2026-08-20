// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fallback execution for row kernels that cannot operate directly on partially valid inputs.
//!
//! This path filters every input from the original row count to the valid row count. It runs the
//! dense kernel on those filtered rows, then scatters the output back to the original row count.

use smallvec::SmallVec;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::super::RowFnExecutionArgs;
use super::super::args::BorrowedRowFnArgs;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::MaskedArray;
use crate::arrays::PrimitiveArray;
use crate::builtins::ArrayBuiltins;
use crate::dtype::Nullability;
use crate::validity::Validity;

impl RowFnExecutionArgs {
    /// Filter the original batch to valid rows, run the dense kernel, then restore its row count.
    pub(super) fn filter_and_scatter(
        &self,
        kernel: impl Fn(BorrowedRowFnArgs<'_>, &mut ExecutionCtx) -> VortexResult<ArrayRef>,
        original_validity: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let original_len = original_validity.len();
        let filtered_len = original_validity.true_count();

        let filtered_inputs: SmallVec<[ArrayRef; 4]> = self
            .inputs
            .iter()
            .map(|input| input.filter(original_validity.clone()))
            .collect::<VortexResult<_>>()?;

        let filtered_args = self.execution_args(&filtered_inputs, filtered_len);
        let filtered = kernel(filtered_args, ctx)?;

        let filtered = self.validate_kernel_output(filtered, filtered_len, ctx)?;

        let output = Self::scatter_to_original_rows(filtered, original_validity)?;

        self.finalize_output(output, original_len)
    }

    /// Scatter `filtered` back to the rows selected by `original_validity`.
    ///
    /// The result has the original length and is null at each invalid position.
    fn scatter_to_original_rows(
        filtered: ArrayRef,
        original_validity: &Mask,
    ) -> VortexResult<ArrayRef> {
        let original_len = original_validity.len();

        let AllOr::Some(valid_slices) = original_validity.slices() else {
            // The caller handles the all-true and all-false masks.
            vortex_bail!(
                "filter-and-scatter requires valid and invalid rows, got an all-valid or all-invalid mask"
            );
        };

        // Map each valid row to its position in `filtered`. Invalid rows use index zero because
        // their gathered values are masked below.
        let mut take_indices = vec![0u64; original_len];

        let valid_rows = valid_slices.iter().flat_map(|&(start, end)| start..end);
        for (filtered_idx, original_idx) in valid_rows.enumerate() {
            take_indices[original_idx] = u64::try_from(filtered_idx)?;
        }

        let take_indices = PrimitiveArray::new(take_indices, Validity::NonNullable).into_array();

        let expanded = filtered.take(take_indices)?;

        // A nullable gathered array cannot be wrapped because a `Masked` child must be all valid.
        // The general masking pass unions its nulls with the batch validity instead.
        if expanded.dtype().is_nullable() {
            let validity_array =
                BoolArray::new(original_validity.to_bit_buffer(), Validity::NonNullable)
                    .into_array();

            return expanded.mask(validity_array);
        }

        // The gathered values are all valid, so attaching validity is sufficient.
        Ok(MaskedArray::try_new(
            expanded,
            Validity::from_mask(original_validity.clone(), Nullability::Nullable),
        )?
        .into_array())
    }
}
