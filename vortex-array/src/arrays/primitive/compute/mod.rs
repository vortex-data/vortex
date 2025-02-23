use crate::arrays::PrimitiveEncoding;
use crate::compute::{
    BetweenFn, CastFn, FillForwardFn, FillNullFn, FilterFn, MaskFn, MinMaxFn, ScalarAtFn,
    SearchSortedFn, SearchSortedUsizeFn, SliceFn, SumFn, TakeFn, ToArrowFn,
};
use crate::vtable::ComputeVTable;
use crate::Array;

mod between;
mod cast;
mod fill;
mod fill_null;
mod filter;
mod mask;
mod min_max;
mod scalar_at;
mod search_sorted;
mod slice;
mod sum;
mod take;
mod to_arrow;

impl ComputeVTable for PrimitiveEncoding {
    fn cast_fn(&self) -> Option<&dyn CastFn<&dyn Array>> {
        Some(self)
    }

    fn between_fn(&self) -> Option<&dyn BetweenFn<&dyn Array>> {
        Some(self)
    }

    fn fill_forward_fn(&self) -> Option<&dyn FillForwardFn<&dyn Array>> {
        Some(self)
    }

    fn fill_null_fn(&self) -> Option<&dyn FillNullFn<&dyn Array>> {
        Some(self)
    }

    fn filter_fn(&self) -> Option<&dyn FilterFn<&dyn Array>> {
        Some(self)
    }

    fn mask_fn(&self) -> Option<&dyn MaskFn<&dyn Array>> {
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

    fn sum_fn(&self) -> Option<&dyn SumFn<&dyn Array>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        Some(self)
    }

    fn to_arrow_fn(&self) -> Option<&dyn ToArrowFn<&dyn Array>> {
        Some(self)
    }

    fn min_max_fn(&self) -> Option<&dyn MinMaxFn<&dyn Array>> {
        Some(self)
    }
}
