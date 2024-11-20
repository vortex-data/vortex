use vortex_array::compute::unary::ScalarAtFn;
use vortex_array::compute::{
    ArrayCompute, ComputeVTable, FilterFn, SearchSortedFn, SliceFn, TakeFn,
};
use vortex_array::ArrayData;

use crate::{BitPackedArray, BitPackedEncoding};

mod filter;
mod scalar_at;
mod search_sorted;
mod slice;
mod take;

impl ArrayCompute for BitPackedArray {
    fn scalar_at(&self) -> Option<&dyn ScalarAtFn> {
        Some(self)
    }

    fn search_sorted(&self) -> Option<&dyn SearchSortedFn> {
        Some(self)
    }

    fn take(&self) -> Option<&dyn TakeFn> {
        Some(self)
    }
}

impl ComputeVTable for BitPackedEncoding {
    fn filter_fn(&self) -> Option<&dyn FilterFn<ArrayData>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<ArrayData>> {
        Some(self)
    }
}
