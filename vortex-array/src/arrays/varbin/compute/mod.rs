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
    fn cast_fn(&self) -> Option<&dyn CastFn<Array>> {
        Some(self)
    }

    fn compare_fn(&self) -> Option<&dyn CompareFn<Array>> {
        Some(self)
    }

    fn filter_fn(&self) -> Option<&dyn FilterFn<Array>> {
        Some(self)
    }

    fn mask_fn(&self) -> Option<&dyn MaskFn<Array>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<Array>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<Array>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<Array>> {
        Some(self)
    }

    fn to_arrow_fn(&self) -> Option<&dyn ToArrowFn<Array>> {
        Some(self)
    }

    fn min_max_fn(&self) -> Option<&dyn MinMaxFn<Array>> {
        Some(self)
    }
}

impl ScalarAtFn<VarBinArray> for VarBinEncoding {
    fn scalar_at(&self, array: &VarBinArray, index: usize) -> VortexResult<Scalar> {
        Ok(varbin_scalar(array.bytes_at(index)?, array.dtype()))
    }
}
