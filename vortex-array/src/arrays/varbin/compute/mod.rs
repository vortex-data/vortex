pub use min_max::compute_min_max;

use crate::Array;
use crate::arrays::VarBinEncoding;
use crate::compute::TakeFn;
use crate::vtable::ComputeVTable;

mod cast;
mod compare;
mod filter;
mod is_constant;
mod is_sorted;
mod mask;
mod min_max;
mod take;

impl ComputeVTable for VarBinEncoding {
    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        Some(self)
    }
}
