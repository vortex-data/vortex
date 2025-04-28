mod between;
mod filter;
mod is_constant;
mod is_sorted;
mod scalar_at;
mod slice;
mod sum;
mod take;
mod to_arrow;

use crate::Array;
use crate::arrays::DecimalEncoding;
use crate::compute::{IsConstantFn, IsSortedFn, ScalarAtFn, SliceFn, TakeFn, ToArrowFn};
use crate::vtable::ComputeVTable;

impl ComputeVTable for DecimalEncoding {
    fn is_constant_fn(&self) -> Option<&dyn IsConstantFn<&dyn Array>> {
        Some(self)
    }

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

    fn to_arrow_fn(&self) -> Option<&dyn ToArrowFn<&dyn Array>> {
        Some(self)
    }
}
