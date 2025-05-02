mod between;
mod filter;
mod is_constant;
mod is_sorted;
mod min_max;
mod scalar_at;
mod sum;
mod take;
mod uncompressed_size;

use crate::Array;
use crate::arrays::DecimalEncoding;
use crate::compute::{IsSortedFn, MinMaxFn, ScalarAtFn, TakeFn, UncompressedSizeFn};
use crate::vtable::ComputeVTable;

impl ComputeVTable for DecimalEncoding {
    fn is_sorted_fn(&self) -> Option<&dyn IsSortedFn<&dyn Array>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<&dyn Array>> {
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
