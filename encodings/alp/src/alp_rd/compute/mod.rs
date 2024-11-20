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

    fn slice(&self) -> Option<&dyn SliceFn> {
        Some(self)
    }

    fn take(&self) -> Option<&dyn TakeFn> {
        Some(self)
    }
}

impl ComputeVTable for ALPRDEncoding {
    fn filter_fn(&self) -> Option<&dyn FilterFn<ArrayData>> {
        Some(self)
    }
}
