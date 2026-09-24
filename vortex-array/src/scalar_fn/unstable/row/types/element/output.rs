// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Owned scalar values that can be collected into all-valid output columns.
//!
//! [`OutputElement`] describes fixed-dtype values returned independently by each row invocation.

use std::mem::MaybeUninit;

use vortex_buffer::BufferAllocatorRef;
use vortex_compute::lane_kernels::IndexedSource;
use vortex_compute::lane_kernels::IndexedSourceExt;

use crate::ArrayRef;
use crate::dtype::DType;

/// An owned row value that can be built into an all-valid column.
///
/// Skip-invalid execution uses [`Default`] only as a placeholder for invalid rows. Batch execution
/// masks those rows before returning the output. Each implementation owns allocation and array
/// construction. Execution only writes through its [`OutputBuffer`] slots.
pub trait OutputElement: 'static + Sized + Default {
    /// Storage used to collect this element type before constructing its column.
    type Buffer: OutputBuffer<Self> + 'static;

    /// The dtype of columns built from this element type. **Must** be non-nullable: nullability is
    /// derived from the inputs by batch execution.
    ///
    /// Because this method takes no arguments, the dtype must be a property of the Rust type. Use
    /// an [`OutputSink`] when the output dtype depends on function options.
    ///
    /// [`OutputSink`]: crate::scalar_fn::unstable::row::OutputSink
    fn element_dtype() -> DType;

    /// Allocate storage with at least `rows` writable slots using the execution allocator.
    ///
    /// Finishing the buffer must produce an all-valid column whose dtype matches
    /// [`element_dtype`](Self::element_dtype) except for outer nullability.
    fn with_capacity(rows: usize, allocator: &BufferAllocatorRef) -> Self::Buffer;

    /// Map a contiguous row source directly into an all-valid column.
    ///
    /// The default writes into the storage returned by [`with_capacity`](Self::with_capacity), then
    /// finishes it.
    /// An output type can override this method when its physical representation supports a more
    /// efficient bulk mapping. The implementation **must** call `apply` exactly once for every
    /// source row in increasing order and return the same values as the default implementation.
    /// Any new payload buffers must use `allocator`.
    ///
    /// An override **must not** introduce value-dependent errors or panics. Fallible operations
    /// must use a fallible visitor path so that [`RowFn::INFALLIBLE`] continues to protect optimizer
    /// transformations.
    ///
    /// [`RowFn::INFALLIBLE`]: crate::scalar_fn::unstable::row::RowFn::INFALLIBLE
    fn build_from<S, F>(source: S, apply: F, allocator: &BufferAllocatorRef) -> ArrayRef
    where
        S: IndexedSource,
        F: Fn(S::Item) -> Self,
    {
        let row_count = source.len();
        let mut values = Self::with_capacity(row_count, allocator);
        let output = &mut values.slots()[..row_count];

        source.map_into(output, apply);

        // SAFETY: normal completion of `map_into` initializes every output slot exactly once.
        unsafe { values.finish(row_count, allocator) }
    }
}

/// Engine-owned storage for collecting independent row values.
///
/// The executor initializes a prefix of [`slots`](Self::slots), then calls [`finish`](Self::finish).
/// Dropping the buffer instead abandons the output, including after an error or unwind.
///
/// # Safety
///
/// - Access through this trait must preserve slot count and contents between calls to
///   [`slots`](Self::slots), including across moves of the buffer.
/// - Dropping the buffer must be safe with any subset of its slots initialized.
///
/// Violating these requirements can cause undefined behavior in an executor or sink.
pub unsafe trait OutputBuffer<T>: Sized {
    /// Writable capacity whose initialized prefix is published by [`finish`](Self::finish).
    fn slots(&mut self) -> &mut [MaybeUninit<T>];

    /// Publish the first `len` slots as an all-valid column, reusing storage where possible.
    ///
    /// The column must contain exactly `len` rows in slot order. Any new payload buffers must
    /// use `allocator`.
    ///
    /// # Safety
    ///
    /// The first `len` slots must exist and contain initialized values. Violating this requirement
    /// can cause undefined behavior.
    unsafe fn finish(self, len: usize, allocator: &BufferAllocatorRef) -> ArrayRef;
}
