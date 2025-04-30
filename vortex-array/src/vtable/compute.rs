use crate::Array;
use crate::compute::{
    FillNullFn, IsSortedFn, LikeFn, MinMaxFn, NaNCountFn, OptimizeFn, ScalarAtFn, SearchSortedFn,
    SearchSortedUsizeFn, SliceFn, TakeFn, TakeFromFn, UncompressedSizeFn,
};

/// VTable for dispatching compute functions to Vortex encodings.
pub trait ComputeVTable {
    /// Fill null values with given desired value. Resulting array is NonNullable
    ///
    /// See: [`FillNullFn`]
    fn fill_null_fn(&self) -> Option<&dyn FillNullFn<&dyn Array>> {
        None
    }

    fn is_sorted_fn(&self) -> Option<&dyn IsSortedFn<&dyn Array>> {
        None
    }

    /// Perform a SQL LIKE operation on two arrays.
    ///
    /// See: [`LikeFn`].
    fn like_fn(&self) -> Option<&dyn LikeFn<&dyn Array>> {
        None
    }

    /// Compute the min, max of an array.
    ///
    /// See: [`MinMaxFn`].
    fn min_max_fn(&self) -> Option<&dyn MinMaxFn<&dyn Array>> {
        None
    }

    /// Compute nan count of the array
    ///
    /// See: [`NaNCountFn`]
    fn nan_count_fn(&self) -> Option<&dyn NaNCountFn<&dyn Array>> {
        None
    }

    /// Try and optimize the layout of an array.
    ///
    /// See: [`OptimizeFn`]
    fn optimize_fn(&self) -> Option<&dyn OptimizeFn<&dyn Array>> {
        None
    }

    /// Single item indexing on Vortex arrays.
    ///
    /// See: [`ScalarAtFn`].
    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<&dyn Array>> {
        None
    }

    /// Perform a search over an ordered array.
    ///
    /// See: [`SearchSortedFn`].
    fn search_sorted_fn(&self) -> Option<&dyn SearchSortedFn<&dyn Array>> {
        None
    }

    /// Perform a search over an ordered array.
    ///
    /// See: [`SearchSortedUsizeFn`].
    fn search_sorted_usize_fn(&self) -> Option<&dyn SearchSortedUsizeFn<&dyn Array>> {
        None
    }

    /// Perform zero-copy slicing of an array.
    ///
    /// See: [`SliceFn`].
    fn slice_fn(&self) -> Option<&dyn SliceFn<&dyn Array>> {
        None
    }

    /// Take a set of indices from an array. This often forces allocations and decoding of
    /// the receiver.
    ///
    /// See: [`TakeFn`].
    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        None
    }

    fn take_from_fn(&self) -> Option<&dyn TakeFromFn<&dyn Array>> {
        None
    }

    /// Approximates the uncompressed size of the array.
    ///
    /// See [`UncompressedSizeFn`]
    fn uncompressed_size_fn(&self) -> Option<&dyn UncompressedSizeFn<&dyn Array>> {
        None
    }
}
