use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::ArrayRef;
use crate::array::Cost;
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

    /// Whether all values in the array are the same.
    ///
    /// The cost parameter determines how much work should be done by this function to reach
    /// that determination. Any recursion into child arrays should therefore be careful to forward
    /// the cost parameter.
    ///
    /// If the const-ness of the array cannot be determined under the cost constraints, then
    /// `false` is returned.
    ///
    /// Computing const-ness with [`Cost::Canonicalize`] will guarantee an exact result.
    ///
    /// ## Preconditions
    ///
    /// * All values are valid
    /// * array.len() > 1
    fn is_constant(_array: &V::Array, _cost: Cost) -> VortexResult<Option<bool>> {
        Ok(None)
    }
}
