use crate::Array;
use crate::arrays::BoolEncoding;
use crate::compute::{TakeFn, UncompressedSizeFn};
use crate::vtable::ComputeVTable;

mod cast;
mod fill_null;
pub mod filter;
mod flatten;
mod invert;
mod is_constant;
mod is_sorted;
mod mask;
mod min_max;
mod sum;
mod take;
mod uncompressed_size;

impl ComputeVTable for BoolEncoding {
    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        Some(self)
    }

    fn uncompressed_size_fn(&self) -> Option<&dyn UncompressedSizeFn<&dyn Array>> {
        Some(self)
    }
}
