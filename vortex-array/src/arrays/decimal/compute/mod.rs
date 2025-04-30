mod between;
mod filter;
mod is_constant;
mod is_sorted;
mod scalar_at;
mod slice;
mod sum;
mod take;

use crate::Array;
use crate::arrays::DecimalEncoding;
use crate::compute::{IsSortedFn, ScalarAtFn, SliceFn, TakeFn};
use crate::vtable::ComputeVTable;

impl ComputeVTable for DecimalEncoding {
    fn is_sorted_fn(&self) -> Option<&dyn IsSortedFn<&dyn Array>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<&dyn Array>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<&dyn Array>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        Some(self)
    }
}
