mod filter;
mod scalar_at;
mod slice;
mod sum;

use crate::arrays::{DecimalArray, DecimalEncoding};
use crate::compute::{FilterKernelAdapter, KernelRef, ScalarAtFn, SliceFn, SumFn};
use crate::vtable::ComputeVTable;
use crate::{Array, ArrayComputeImpl};

impl ArrayComputeImpl for DecimalArray {
    const FILTER: Option<KernelRef> = FilterKernelAdapter(DecimalEncoding).some();
}

impl ComputeVTable for DecimalEncoding {
    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<&dyn Array>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<&dyn Array>> {
        Some(self)
    }

    fn sum_fn(&self) -> Option<&dyn SumFn<&dyn Array>> {
        Some(self)
    }

    // TODO(aduffy): BetweenFn
    // TODO(aduffy): IsSortedFn
    // TODO(aduffy): SearchSortedFn
    // TODO(aduffy): CompareFn
    // TODO(aduffy): IsConstant
    // TODO(aduffy): BetweenFn
    // TODO(aduffy): BinaryNumericFn
    // TODO(aduffy): TakeFn
}
