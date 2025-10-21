// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::vortex_panic;
use vortex_scalar::Scalar;

use crate::ArrayRef;
use crate::vtable::{NotSupported, VTable};

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
    fn slice(array: &V::Array, range: Range<usize>) -> ArrayRef;

    /// Fetch the scalar at the given index.
    ///
    /// ## Preconditions
    ///
    /// Bounds-checking has already been performed by the time this function is called,
    /// and the index is guaranteed to be non-null.
    fn scalar_at(array: &V::Array, index: usize) -> Scalar;
}

impl<V: VTable> OperationsVTable<V> for NotSupported {
    fn slice(array: &V::Array, _range: Range<usize>) -> ArrayRef {
        vortex_panic!(
            "Legacy slice operation is not supported for {} arrays",
            array.encoding_id()
        )
    }

    fn scalar_at(array: &V::Array, _index: usize) -> Scalar {
        vortex_panic!(
            "Legacy scalar_at operation is not supported for {} arrays",
            array.encoding_id()
        )
    }
}
