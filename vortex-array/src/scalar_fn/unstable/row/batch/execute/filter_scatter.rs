// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fallback execution for row kernels that cannot operate on the original partially valid inputs.
//!
//! Direct skip-invalid execution retains the original row domain: it decodes every input column,
//! initializes a full-length output, and visits only valid rows. Some input representations cannot
//! decode unspecified payloads behind nulls, and some output sinks cannot initialize rows that the
//! kernel skips. In either case, there is no safe row loop to enter over the original columns.
//!
//! This fallback filters every input to the valid row domain before decoding it. The ordinary
//! kernel then produces a compact, all-valid output. A take with nullable indices scatters those
//! values back to the original row domain and restores its nulls.
//!
//! Filtering and scattering add columnar work around the row loop. Filtering nested or compressed
//! inputs can also materialize their selected representation. Batch execution therefore tries
//! direct skip-invalid execution first and uses this path only when a required capability is
//! unavailable.
//!
//! A prepared, type-specific view can provide selected-row decoding without filtering. That is an
//! optional direct-execution capability rather than a replacement for this fallback: it requires a
//! representation with safe selected access, and it does not initialize an output sink that cannot
//! represent skipped rows. Calling the general scalar-extraction API once per valid row is also not
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
use crate::arrays::PrimitiveArray;
use crate::dtype::Nullability;
use crate::validity::Validity;

impl RowFnExecutionArgs {
    /// Filter the original batch to valid rows, run the kernel, then restore its row count.
    pub(super) fn filter_and_scatter(
        &self,
        kernel: impl Fn(BorrowedRowFnArgs<'_>, &mut ExecutionCtx) -> VortexResult<ArrayRef>,
        original_validity: &MaskValuesRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let original_len = original_validity.len();
        let filtered_len = original_validity.true_count();
        let filter_mask = Mask::Values(MaskValuesRef::clone(original_validity));

        let filtered_inputs: SmallVec<[ArrayRef; 4]> = self
            .inputs
            .iter()
            .map(|input| input.filter(filter_mask.clone()))
            .collect::<VortexResult<_>>()?;

        let filtered_args = self.execution_args(&filtered_inputs, filtered_len);
        let filtered = kernel(filtered_args, ctx)?;
        let filtered = self.validate_kernel_output(filtered, filtered_len, ctx)?;

        let output = Self::scatter_to_original_rows(filtered, original_validity)?;

        self.finalize_output(output, original_len)
    }

    /// Scatter `filtered` back to the rows selected by `original_validity`.
    fn scatter_to_original_rows(
        filtered: ArrayRef,
        original_validity: &MaskValuesRef,
    ) -> VortexResult<ArrayRef> {
        let original_len = original_validity.len();
        let mut take_indices = vec![0u64; original_len];

        let valid_rows = original_validity
            .slices()
            .iter()
            .flat_map(|&(start, end)| start..end);
        for (filtered_index, original_index) in valid_rows.enumerate() {
            take_indices[original_index] = u64::try_from(filtered_index)?;
        }

        // Null indices restore invalid rows without selecting a value from the compact output.
        let take_indices = PrimitiveArray::new(
            take_indices,
            Validity::from_mask(
                Mask::Values(MaskValuesRef::clone(original_validity)),
                Nullability::Nullable,
            ),
        )
        .into_array();

        filtered.take(take_indices)
    }
}
