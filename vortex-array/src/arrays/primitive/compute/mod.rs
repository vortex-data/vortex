use crate::Array;
use crate::arrays::PrimitiveEncoding;
use crate::compute::{SearchSortedFn, SearchSortedUsizeFn};
use crate::vtable::ComputeVTable;

mod between;
mod cast;
mod fill_null;
mod filter;
mod is_constant;
mod is_sorted;
mod mask;
mod min_max;
mod nan_count;
mod search_sorted;
mod sum;
mod take;

pub use is_constant::*;

impl ComputeVTable for PrimitiveEncoding {
    fn search_sorted_fn(&self) -> Option<&dyn SearchSortedFn<&dyn Array>> {
        Some(self)
    }

    fn search_sorted_usize_fn(&self) -> Option<&dyn SearchSortedUsizeFn<&dyn Array>> {
        Some(self)
    }
}
