use vortex_array::Array;
use vortex_array::compute::{ScalarAtFn, TakeFn};
use vortex_array::vtable::ComputeVTable;

use crate::ALPRDEncoding;

mod filter;
mod mask;
mod scalar_at;
mod take;

impl ComputeVTable for ALPRDEncoding {
    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<&dyn Array>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        Some(self)
    }
}
