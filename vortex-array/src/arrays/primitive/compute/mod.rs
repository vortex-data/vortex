use crate::Array;
use crate::arrays::PrimitiveEncoding;
use crate::compute::{
    FillNullFn, IsConstantFn, IsSortedFn, MaskFn, MinMaxFn, ScalarAtFn, SearchSortedFn,
    SearchSortedUsizeFn, SliceFn, TakeFn, ToArrowFn, UncompressedSizeFn,
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
mod scalar_at;
mod search_sorted;
mod slice;
mod sum;
mod take;
mod to_arrow;
mod uncompressed_size;

pub use is_constant::*;

impl ComputeVTable for PrimitiveEncoding {
    fn fill_null_fn(&self) -> Option<&dyn FillNullFn<&dyn Array>> {
        Some(self)
    }

    fn is_constant_fn(&self) -> Option<&dyn IsConstantFn<&dyn Array>> {
        Some(self)
    }

    fn is_sorted_fn(&self) -> Option<&dyn IsSortedFn<&dyn Array>> {
        Some(self)
    }

    fn mask_fn(&self) -> Option<&dyn MaskFn<&dyn Array>> {
        Some(self)
    }

    fn min_max_fn(&self) -> Option<&dyn MinMaxFn<&dyn Array>> {
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

    fn slice_fn(&self) -> Option<&dyn SliceFn<&dyn Array>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        Some(self)
    }

    fn to_arrow_fn(&self) -> Option<&dyn ToArrowFn<&dyn Array>> {
        Some(self)
    }

    fn uncompressed_size_fn(&self) -> Option<&dyn UncompressedSizeFn<&dyn Array>> {
        Some(self)
    }
}
