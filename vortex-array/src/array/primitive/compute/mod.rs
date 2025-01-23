use crate::array::PrimitiveEncoding;
use crate::compute::{
    CastFn, ComputeVTable, FillForwardFn, FillNullFn, FilterFn, ScalarAtFn, SearchSortedFn,
    SearchSortedUsizeFn, SliceFn, TakeFn,
};
use crate::ArrayData;

mod cast;
mod fill;
mod fill_null;
mod filter;
mod scalar_at;
mod search_sorted;
mod slice;
mod take;

impl ComputeVTable for PrimitiveEncoding {
    fn cast_fn(&self) -> Option<&dyn CastFn<ArrayData>> {
        Some(self)
    }

    fn fill_forward_fn(&self) -> Option<&dyn FillForwardFn<ArrayData>> {
        Some(self)
    }

    fn fill_null_fn(&self) -> Option<&dyn FillNullFn<ArrayData>> {
        Some(self)
    }

    fn filter_fn(&self) -> Option<&dyn FilterFn<ArrayData>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<ArrayData>> {
        Some(self)
    }

    fn search_sorted_fn(&self) -> Option<&dyn SearchSortedFn<ArrayData>> {
        Some(self)
    }

    fn search_sorted_usize_fn(&self) -> Option<&dyn SearchSortedUsizeFn<ArrayData>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<ArrayData>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<ArrayData>> {
        Some(self)
    }
}
