use vortex_array::compute::{FilterKernelAdapter, KernelRef, MaskFn, ScalarAtFn, SliceFn, TakeFn};
use vortex_array::vtable::ComputeVTable;
use vortex_array::{Array, ArrayComputeImpl};

use crate::{ALPRDArray, ALPRDEncoding};

mod filter;
mod mask;
mod scalar_at;
mod slice;
mod take;

impl ArrayComputeImpl for ALPRDArray {
    const FILTER: Option<KernelRef> = FilterKernelAdapter(ALPRDEncoding).some();
}

impl ComputeVTable for ALPRDEncoding {
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
