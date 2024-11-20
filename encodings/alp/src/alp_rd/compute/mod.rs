use vortex_array::compute::unary::ScalarAtFn;
use vortex_array::compute::{ArrayCompute, ComputeVTable, FilterFn, SliceFn, TakeFn};
use vortex_array::ArrayData;

use crate::{ALPRDArray, ALPRDEncoding};

mod filter;
mod scalar_at;
mod slice;
mod take;

impl ArrayCompute for ALPRDArray {
    fn scalar_at(&self) -> Option<&dyn ScalarAtFn> {
        Some(self)
    }
}

impl ComputeVTable for ALPRDEncoding {
    fn filter_fn(&self) -> Option<&dyn FilterFn<ArrayData>> {
        Some(self)
    }
    fn slice_fn(&self) -> Option<&dyn SliceFn<ArrayData>> {
        Some(self)
    }
    fn take_fn(&self) -> Option<&dyn TakeFn<ArrayData>> {
        Some(self)
    }
}
