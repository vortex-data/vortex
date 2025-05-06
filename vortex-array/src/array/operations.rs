use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::ArrayRef;

/// Implementation trait for common or required array operations.
///
/// There is no good answer for what should be an extensible compute function, vs a built-in
/// operation. At the moment, we promote a compute function to a built-in operation if it is
/// called sufficiently often that the compute dispatch overhead is non-trivial.
pub trait ArrayOperationsImpl {
    /// Perform a constant-time slice of the array.
    ///
    /// If an encoding cannot perform this slice in constant time, it should internally
    /// store an offset and length in order to defer slicing until the array is accessed.
    ///
    /// Note that bounds-checking has already been performed by the time this function is called.
    fn _slice(&self, start: usize, stop: usize) -> VortexResult<ArrayRef>;

    /// Fetch the scalar at the given index.
    ///
    /// Note that bounds-checking has already been performed by the time this function is called,
    /// and the index is guaranteed to be valid.
    fn _scalar_at(&self, index: usize) -> VortexResult<Scalar>;

    // TODO(ngates): add _is_constant here
}
