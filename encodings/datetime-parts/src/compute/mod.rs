mod cast;
mod compare;
mod filter;
mod is_constant;
mod take;

use vortex_array::Array;
use vortex_array::compute::TakeFn;
use vortex_array::vtable::ComputeVTable;

use crate::DateTimePartsEncoding;

impl ComputeVTable for DateTimePartsEncoding {
    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        Some(self)
    }

    // TODO(joe): implement `between_fn` this is used at lot.
}
