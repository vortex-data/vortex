use crate::Array;
use crate::compute::{SearchSortedFn, SearchSortedUsizeFn};

/// VTable for dispatching compute functions to Vortex encodings.
pub trait ComputeVTable {
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
}
