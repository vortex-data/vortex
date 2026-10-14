// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Borrowed access to the input columns of one row-function execution.
//!
//! [`RowInputs`] lets batch execution inspect and decode its inputs without cloning them. An
//! internal caller that holds its operands implements it directly. [`ArrayInputs`] adapts the
//! columns collected from a general [`ExecutionArgs`](crate::scalar_fn::ExecutionArgs).

use crate::ArrayRef;

/// The same-length input columns of one row-function execution, borrowed by index.
///
/// The trait is `pub` only so that the sealed tuple decoding can bound on it. The private module
/// keeps it out of the public API.
pub trait RowInputs {
    /// Return the number of input columns.
    fn num_inputs(&self) -> usize;

    /// Return the number of rows in the execution scope.
    fn row_count(&self) -> usize;

    /// Return input `index`. `index` **must** be less than [`num_inputs`](Self::num_inputs).
    fn input(&self, index: usize) -> &ArrayRef;
}

/// [`RowInputs`] over a slice of collected columns.
pub(crate) struct ArrayInputs<'a> {
    /// The input columns.
    arrays: &'a [ArrayRef],

    /// The number of rows in the execution scope.
    row_count: usize,
}

impl<'a> ArrayInputs<'a> {
    pub(crate) fn new(arrays: &'a [ArrayRef], row_count: usize) -> Self {
        Self { arrays, row_count }
    }
}

impl RowInputs for ArrayInputs<'_> {
    fn num_inputs(&self) -> usize {
        self.arrays.len()
    }

    fn row_count(&self) -> usize {
        self.row_count
    }

    fn input(&self, index: usize) -> &ArrayRef {
        &self.arrays[index]
    }
}
