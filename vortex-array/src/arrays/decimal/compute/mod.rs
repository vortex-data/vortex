mod between;
mod filter;
mod is_constant;
mod is_sorted;
mod min_max;
mod sum;
mod take;
mod uncompressed_size;

use crate::Array;
use crate::arrays::DecimalEncoding;
use crate::compute::{TakeFn, UncompressedSizeFn};
use crate::vtable::ComputeVTable;

impl ComputeVTable for DecimalEncoding {
    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        Some(self)
    }

    fn uncompressed_size_fn(&self) -> Option<&dyn UncompressedSizeFn<&dyn Array>> {
        Some(self)
    }
}
