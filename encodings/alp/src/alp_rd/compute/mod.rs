use vortex_array::Array;
use vortex_array::compute::{FilterFn, MaskFn, ScalarAtFn, SliceFn, TakeFn};
use vortex_array::vtable::ComputeVTable;

use crate::ALPRDEncoding;

mod filter;
mod mask;
mod scalar_at;
mod slice;
mod take;

impl ComputeVTable for ALPRDEncoding {
    fn filter_fn(&self) -> Option<&dyn FilterFn<&dyn Array>> {
        Some(self)
    }

    fn mask_fn(&self) -> Option<&dyn MaskFn<&dyn Array>> {
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
