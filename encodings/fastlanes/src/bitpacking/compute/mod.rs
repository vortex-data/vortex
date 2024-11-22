use vortex_array::compute::unary::ScalarAtFn;
use vortex_array::compute::{ComputeVTable, FilterFn, SearchSortedFn, SliceFn, TakeFn};
use vortex_array::ArrayData;

use crate::BitPackedEncoding;

mod filter;
mod scalar_at;
mod search_sorted;
mod slice;
mod take;

impl ComputeVTable for BitPackedEncoding {
    fn filter_fn(&self) -> Option<&dyn FilterFn<ArrayData>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<ArrayData>> {
        Some(self)
    }

    fn search_sorted_fn(&self) -> Option<&dyn SearchSortedFn<ArrayData>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<ArrayData>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<ArrayData>> {
        Some(self)
    }
}
