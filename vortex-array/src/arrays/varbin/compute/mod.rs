pub use min_max::compute_min_max;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::Array;
use crate::arrays::VarBinEncoding;
use crate::arrays::varbin::{VarBinArray, varbin_scalar};
use crate::compute::{
    IsSortedFn, MinMaxFn, ScalarAtFn, SliceFn, TakeFn, ToArrowFn, UncompressedSizeFn,
};
use crate::vtable::ComputeVTable;

mod cast;
mod compare;
mod filter;
mod is_constant;
mod is_sorted;
mod mask;
mod min_max;
mod slice;
mod take;
pub(crate) mod to_arrow;
mod uncompressed_size;

impl ComputeVTable for VarBinEncoding {
    fn is_sorted_fn(&self) -> Option<&dyn IsSortedFn<&dyn Array>> {
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

    fn to_arrow_fn(&self) -> Option<&dyn ToArrowFn<&dyn Array>> {
        Some(self)
    }

    fn min_max_fn(&self) -> Option<&dyn MinMaxFn<&dyn Array>> {
        Some(self)
    }

    fn uncompressed_size_fn(&self) -> Option<&dyn UncompressedSizeFn<&dyn Array>> {
        Some(self)
    }
}

impl ScalarAtFn<&VarBinArray> for VarBinEncoding {
    fn scalar_at(&self, array: &VarBinArray, index: usize) -> VortexResult<Scalar> {
        Ok(varbin_scalar(array.bytes_at(index)?, array.dtype()))
    }
}
