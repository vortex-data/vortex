use crate::array::PrimitiveEncoding;
use crate::compute::{
    CastFn, FillForwardFn, FillNullFn, FilterFn, MinMaxFn, ScalarAtFn, SearchSortedFn,
    SearchSortedUsizeFn, SliceFn, TakeFn, ToArrowFn,
};
use crate::vtable::ComputeVTable;
use crate::Array;

mod cast;
mod fill;
mod fill_null;
mod filter;
mod min_max;
mod scalar_at;
mod search_sorted;
mod slice;
mod take;
mod to_arrow;

impl ComputeVTable for PrimitiveEncoding {
    fn cast_fn(&self) -> Option<&dyn CastFn<Array>> {
        Some(self)
    }

    fn fill_forward_fn(&self) -> Option<&dyn FillForwardFn<Array>> {
        Some(self)
    }

    fn fill_null_fn(&self) -> Option<&dyn FillNullFn<Array>> {
        Some(self)
    }

    fn filter_fn(&self) -> Option<&dyn FilterFn<Array>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<Array>> {
        Some(self)
    }

    fn search_sorted_fn(&self) -> Option<&dyn SearchSortedFn<Array>> {
        Some(self)
    }

    fn search_sorted_usize_fn(&self) -> Option<&dyn SearchSortedUsizeFn<Array>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<Array>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<Array>> {
        Some(self)
    }

    fn to_arrow_fn(&self) -> Option<&dyn ToArrowFn<Array>> {
        Some(self)
    }

    fn min_max_fn(&self) -> Option<&dyn MinMaxFn<Array>> {
        Some(self)
    }
}
