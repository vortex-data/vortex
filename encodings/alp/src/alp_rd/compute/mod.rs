use vortex_array::compute::unary::ScalarAtFn;
use vortex_array::compute::{ArrayCompute, FilterFn, SliceFn, TakeFn};

use crate::ALPRDArray;

mod filter;
mod scalar_at;
mod slice;
mod take;

impl ArrayCompute for ALPRDArray {
    fn filter(&self) -> Option<&dyn FilterFn> {
        Some(self)
    }

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
