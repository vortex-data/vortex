use crate::Array;
use crate::compute::SearchSortedFn;

/// VTable for dispatching compute functions to Vortex encodings.
pub trait ComputeVTable {
    /// Perform a search over an ordered array.
    ///
    /// See: [`SearchSortedFn`].
    fn search_sorted_fn(&self) -> Option<&dyn SearchSortedFn<&dyn Array>> {
        None
    }
}
