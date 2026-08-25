// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Owned scalar values that can be collected into all-valid output columns.
//!
//! [`OutputElement`] describes fixed-dtype values returned independently by each row invocation.

use crate::ArrayRef;
use crate::dtype::DType;

/// An owned row value that can be built into an all-valid column.
///
/// Skip-invalid execution uses [`Default`] only as a placeholder for invalid rows. Batch execution
/// masks those rows before returning the output.
pub trait OutputElement: 'static + Sized + Default {
    /// The dtype of columns built from this element type. **Must** be non-nullable: nullability is
    /// derived from the inputs by batch execution.
    ///
    /// Because this method takes no arguments, the dtype must be a property of the Rust type. Use
    /// an [`OutputSink`] when the output dtype depends on function options.
    ///
    /// [`OutputSink`]: crate::scalar_fn::unstable::row::OutputSink
    fn element_dtype() -> DType;

    /// Build an all-valid column from one value per row.
    ///
    /// The returned column must contain `values.len()` rows and match
    /// [`element_dtype`](Self::element_dtype) except for outer nullability. The framework calls
    /// this method once per batch.
    fn build(values: Vec<Self>) -> ArrayRef;
}
