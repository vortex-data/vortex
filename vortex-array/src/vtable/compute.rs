use crate::Array;
use crate::compute::{NaNCountFn, SearchSortedFn, SearchSortedUsizeFn, TakeFn, TakeFromFn};

/// VTable for dispatching compute functions to Vortex encodings.
pub trait ComputeVTable {
    /// Compute nan count of the array
    ///
    /// See: [`NaNCountFn`]
    fn nan_count_fn(&self) -> Option<&dyn NaNCountFn<&dyn Array>> {
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
}
