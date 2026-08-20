// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Owned scalar values that can be collected into all-valid output columns.
//!
//! [`OutputElement`] describes fixed-dtype values returned independently by each row invocation.

use crate::ArrayRef;
use crate::dtype::DType;

/// An owned row value that can be built into an all-valid column.
pub trait OutputElement: 'static + Sized {
    /// The dtype of columns built from this element type. **Must** be non-nullable: nullability is
    /// derived from the inputs by batch execution.
    ///
    /// Because this method takes no arguments, the dtype must be a property of the Rust type. Use
    /// an [`OutputSink`] when the output dtype depends on function options.
    ///
    /// [`OutputSink`]: crate::scalar_fn::unstable::row::OutputSink
    fn element_dtype() -> DType;

    /// Build a column from one value per row. Called once per batch.
    fn build(values: Vec<Self>) -> ArrayRef;
}
