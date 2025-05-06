mod between;
mod filter;
mod is_constant;
mod is_sorted;
mod min_max;
mod sum;
mod take;

use crate::Array;
use crate::arrays::DecimalEncoding;
use crate::compute::TakeFn;
use crate::vtable::ComputeVTable;

impl ComputeVTable for DecimalEncoding {
    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        Some(self)
    }
}
