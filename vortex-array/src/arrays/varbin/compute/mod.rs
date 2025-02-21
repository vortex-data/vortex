pub use min_max::compute_min_max;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::arrays::varbin::{varbin_scalar, VarBinArray};
use crate::arrays::VarBinEncoding;
use crate::compute::{
    CastFn, CompareFn, FilterFn, MaskFn, MinMaxFn, ScalarAtFn, SliceFn, TakeFn, ToArrowFn,
};
use crate::vtable::ComputeVTable;
use crate::Array;

mod cast;
mod compare;
mod filter;
mod mask;
mod min_max;
mod slice;
mod take;
pub(crate) mod to_arrow;

impl ComputeVTable for VarBinEncoding {
    fn cast_fn(&self) -> Option<&dyn CastFn<&dyn Array>> {
        Some(self)
    }

    fn compare_fn(&self) -> Option<&dyn CompareFn<&dyn Array>> {
        Some(self)
    }

    fn filter_fn(&self) -> Option<&dyn FilterFn<&dyn Array>> {
        Some(self)
    }

    fn mask_fn(&self) -> Option<&dyn MaskFn<&dyn Array>> {
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
}

impl ScalarAtFn<&VarBinArray> for VarBinEncoding {
    fn scalar_at(&self, array: &VarBinArray, index: usize) -> VortexResult<Scalar> {
        Ok(varbin_scalar(array.bytes_at(index)?, array.dtype()))
    }
}
