// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution arguments paired with the metadata selected during planning.
//!
//! [`BorrowedRowFnArgs`] can point at original or sliced arrays while retaining the dtypes and
//! [`BatchPlan`] of the original batch.

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::BatchPlan;
use crate::ArrayRef;
use crate::dtype::DType;
use crate::scalar_fn::ExecutionArgs;

/// A borrowed [`ExecutionArgs`] view with the metadata selected for its row function.
///
/// `arrays` can be sliced, while `dtypes` and `plan` always describe the original planned batch.
/// Keeping them together prevents an execution path from pairing an input view with unrelated
/// planning metadata.
#[derive(Clone, Copy)]
pub(crate) struct BorrowedRowFnArgs<'a> {
    /// The input arrays for this row-function invocation.
    arrays: &'a [ArrayRef],

    /// The number of rows in this row-function invocation.
    row_count: usize,

    /// The original input dtypes used to select the row implementation.
    dtypes: &'a [DType],

    /// The plan an executing dispatch must reproduce.
    plan: &'a BatchPlan,
}

impl<'a> BorrowedRowFnArgs<'a> {
    /// Pair one input view with the planning metadata selected for its batch.
    pub(crate) fn new(
        arrays: &'a [ArrayRef],
        row_count: usize,
        dtypes: &'a [DType],
        plan: &'a BatchPlan,
    ) -> Self {
        Self {
            arrays,
            row_count,
            dtypes,
            plan,
        }
    }

    /// Return the original input dtypes used to select the row implementation.
    pub(crate) fn dtypes(&self) -> &'a [DType] {
        self.dtypes
    }

    /// Return the plan an executing dispatch must reproduce.
    pub(crate) fn plan(&self) -> &'a BatchPlan {
        self.plan
    }
}

impl ExecutionArgs for BorrowedRowFnArgs<'_> {
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
