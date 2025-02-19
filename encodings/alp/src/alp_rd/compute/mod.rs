use vortex_array::compute::{FilterFn, MaskFn, ScalarAtFn, SliceFn, TakeFn};
use vortex_array::vtable::ComputeVTable;
use vortex_array::ArrayRef;

use crate::ALPRDEncoding;

mod filter;
mod mask;
mod scalar_at;
mod slice;
mod take;

impl ComputeVTable for ALPRDEncoding {
    fn filter_fn(&self) -> Option<&dyn FilterFn<ArrayRef>> {
        Some(self)
    }

    fn mask_fn(&self) -> Option<&dyn MaskFn<ArrayRef>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<ArrayRef>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<ArrayRef>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<ArrayRef>> {
        Some(self)
    }
}
