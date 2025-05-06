use crate::Array;
use crate::arrays::PrimitiveEncoding;
use crate::compute::{
    NaNCountFn, ScalarAtFn, SearchSortedFn, SearchSortedUsizeFn, TakeFn, UncompressedSizeFn,
};
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
mod scalar_at;
mod search_sorted;
mod sum;
mod take;
mod uncompressed_size;

pub use is_constant::*;

impl ComputeVTable for PrimitiveEncoding {
    fn nan_count_fn(&self) -> Option<&dyn NaNCountFn<&dyn Array>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<&dyn Array>> {
        Some(self)
    }

    fn search_sorted_fn(&self) -> Option<&dyn SearchSortedFn<&dyn Array>> {
        Some(self)
    }

    fn search_sorted_usize_fn(&self) -> Option<&dyn SearchSortedUsizeFn<&dyn Array>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        Some(self)
    }

    fn uncompressed_size_fn(&self) -> Option<&dyn UncompressedSizeFn<&dyn Array>> {
        Some(self)
    }
}
