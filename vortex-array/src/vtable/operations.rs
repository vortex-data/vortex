// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::ArrayRef;
use crate::vtable::VTable;

pub trait OperationsVTable<V: VTable> {
    /// Perform a constant-time slice of the array.
    ///
    /// If an encoding cannot perform this slice in constant time, it should internally
    /// store an offset and length in order to defer slicing until the array is accessed.
    ///
    /// This function returns [`ArrayRef`] since some encodings can return a simpler array for
    /// some slices, for example a [`crate::arrays::ChunkedArray`] may slice into a single chunk.
    ///
    /// ## Preconditions
    ///
    /// Bounds-checking has already been performed by the time this function is called.
    fn slice(array: &V::Array, start: usize, stop: usize) -> VortexResult<ArrayRef>;

    /// Fetch the scalar at the given index.
    ///
    /// ## Preconditions
    ///
    /// Bounds-checking has already been performed by the time this function is called,
    /// and the index is guaranteed to be non-null.
    fn scalar_at(array: &V::Array, index: usize) -> VortexResult<Scalar>;

    /// Return an optimized copy of an Array, with any unreferenced data blocks freed and
    /// any extraneous information removed.
    ///
    /// Many simple contiguous array types do not benefit from this operation, but it is
    /// especially useful for variable-length types which keep a variable number of buffers
    /// that can be dereferenced by other internal data structures via simple zero-copy
    /// filter and take operations. After multiple operations are applied to these arrays, it is
    /// common for the majority of owned buffer data to no longer logically be referenced.
    ///
    /// This operation can be called to return a new copy of an array with the same encoding,
    /// but with all unreferenced data unlinked.
    ///
    /// ## Default behavior
    ///
    /// For most arrays that do not contain variable buffer counts, such as the canonical
    /// arrays, the default implementation will not attempt to perform compaction and instead
    /// return the original array.
    fn optimize(array: &V::Array) -> VortexResult<ArrayRef> {
        Ok(array.to_array())
    }
}
