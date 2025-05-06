use vortex_array::Array;
use vortex_array::compute::TakeFn;
use vortex_array::vtable::ComputeVTable;

use crate::ALPRDEncoding;

mod filter;
mod mask;
mod take;

impl ComputeVTable for ALPRDEncoding {
    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        Some(self)
    }
}
