use vortex_array::compute::{FilterFn, ScalarAtFn, SliceFn, TakeFn};
use vortex_array::vtable::ComputeVTable;
use vortex_array::Array;

use crate::ALPRDEncoding;

mod filter;
mod scalar_at;
mod slice;
mod take;

impl ComputeVTable for ALPRDEncoding {
    fn filter_fn(&self) -> Option<&dyn FilterFn<Array>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<Array>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<Array>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<Array>> {
        Some(self)
    }
}
