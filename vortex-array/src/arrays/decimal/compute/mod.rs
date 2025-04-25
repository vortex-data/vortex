mod between;
mod filter;
mod is_constant;
mod is_sorted;
mod scalar_at;
mod slice;
mod sum;
mod take;

use crate::arrays::{DecimalArray, DecimalEncoding};
use crate::compute::{
    BetweenFn, FilterKernelAdapter, IsConstantFn, IsSortedFn, KernelRef, ScalarAtFn, SliceFn,
    SumFn, TakeFn,
};
use crate::vtable::ComputeVTable;
use crate::{Array, ArrayComputeImpl};

impl ArrayComputeImpl for DecimalArray {
    const FILTER: Option<KernelRef> = FilterKernelAdapter(DecimalEncoding).some();
}

impl ComputeVTable for DecimalEncoding {
    fn between_fn(&self) -> Option<&dyn BetweenFn<&dyn Array>> {
        Some(self)
    }

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

    fn sum_fn(&self) -> Option<&dyn SumFn<&dyn Array>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        Some(self)
    }
}
