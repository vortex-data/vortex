// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution arguments paired with the metadata selected during planning.
//!
//! [`BorrowedExecutionArgs`] can point at original or sliced arrays while retaining the dtypes,
//! output dtype, and execution policy of the original batch plan.

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::RowPolicy;
use crate::ArrayRef;
use crate::dtype::DType;
use crate::scalar_fn::ExecutionArgs;

/// A borrowed [`ExecutionArgs`] view with the metadata selected for its row function.
///
/// `arrays` can be sliced, while `dtypes` and `output_dtype` always describe the original planned
/// batch. Keeping them together prevents an execution path from pairing an input view with
/// unrelated planning metadata.
#[derive(Clone, Copy)]
pub(crate) struct BorrowedExecutionArgs<'a> {
    /// The input arrays for this row-function invocation.
    arrays: &'a [ArrayRef],

    /// The number of rows in this row-function invocation.
    row_count: usize,

    /// The original input dtypes used to select the row implementation.
    dtypes: &'a [DType],

    /// The non-nullable dtype built by the selected output capability.
    output_dtype: &'a DType,

    /// The nullable execution policy selected during planning.
    policy: RowPolicy,
}

impl<'a> BorrowedExecutionArgs<'a> {
    /// Pair one input view with the planning metadata selected for its batch.
    pub(crate) fn new(
        arrays: &'a [ArrayRef],
        row_count: usize,
        dtypes: &'a [DType],
        output_dtype: &'a DType,
        policy: RowPolicy,
    ) -> Self {
        Self {
            arrays,
            row_count,
            dtypes,
            output_dtype,
            policy,
        }
    }

    /// Return the original input dtypes used to select the row implementation.
    pub(crate) fn dtypes(&self) -> &'a [DType] {
        self.dtypes
    }

    /// Return the non-nullable dtype built by the selected output capability.
    pub(crate) fn output_dtype(&self) -> &'a DType {
        self.output_dtype
    }

    /// Return the nullable execution policy selected during planning.
    pub(crate) fn policy(&self) -> RowPolicy {
        self.policy
    }
}

impl ExecutionArgs for BorrowedExecutionArgs<'_> {
    fn get(&self, index: usize) -> VortexResult<ArrayRef> {
        self.arrays.get(index).cloned().ok_or_else(|| {
            vortex_err!(
                "row-function input index must be less than {}, got {index}",
                self.arrays.len(),
            )
        })
    }

    fn num_inputs(&self) -> usize {
        self.arrays.len()
    }

    fn row_count(&self) -> usize {
        self.row_count
    }
}
