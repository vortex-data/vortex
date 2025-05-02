pub use min_max::compute_min_max;

use crate::Array;
use crate::arrays::VarBinEncoding;
use crate::compute::{IsSortedFn, MinMaxFn, TakeFn, UncompressedSizeFn};
use crate::vtable::ComputeVTable;

mod cast;
mod compare;
mod filter;
mod is_constant;
mod is_sorted;
mod mask;
mod min_max;
mod take;
mod uncompressed_size;

impl ComputeVTable for VarBinEncoding {
    fn is_sorted_fn(&self) -> Option<&dyn IsSortedFn<&dyn Array>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        Some(self)
    }

    fn min_max_fn(&self) -> Option<&dyn MinMaxFn<&dyn Array>> {
        Some(self)
    }

    fn uncompressed_size_fn(&self) -> Option<&dyn UncompressedSizeFn<&dyn Array>> {
        Some(self)
    }
}
