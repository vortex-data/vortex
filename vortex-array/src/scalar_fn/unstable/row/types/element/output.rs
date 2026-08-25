// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Owned scalar values that can be collected into all-valid output columns.
//!
//! [`OutputElement`] describes fixed-dtype values returned independently by each row invocation.

use vortex_compute::lane_kernels::IndexedSource;
use vortex_compute::lane_kernels::IndexedSourceExt;

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
    /// [`element_dtype`](Self::element_dtype) except for outer nullability. The default
    /// [`build_from`](Self::build_from) implementation and valid-row execution call this method.
    fn build(values: Vec<Self>) -> ArrayRef;

    /// Map a contiguous row source directly into an all-valid column.
    ///
    /// The default collects one value per row into a [`Vec`] before calling [`build`](Self::build).
    /// An output type can override this method when its physical representation supports a more
    /// efficient bulk mapping. The implementation **must** call `apply` exactly once for every
    /// source row in increasing order and return the same values as the default implementation.
    ///
    /// An override **must not** introduce value-dependent errors or panics. Fallible operations
    /// must use a fallible visitor path so that [`RowFn::INFALLIBLE`] continues to protect optimizer
    /// transformations.
    ///
    /// [`RowFn::INFALLIBLE`]: crate::scalar_fn::unstable::row::RowFn::INFALLIBLE
    fn build_from<S, F>(source: S, apply: F) -> ArrayRef
    where
        S: IndexedSource,
        F: Fn(S::Item) -> Self,
    {
        let row_count = source.len();
        let mut values = Vec::<Self>::with_capacity(row_count);
        let output = &mut values.spare_capacity_mut()[..row_count];

        source.map_into(output, apply);

        // SAFETY: normal completion of `map_into` initializes every output slot exactly once.
        unsafe { values.set_len(row_count) };

        Self::build(values)
    }
}
