use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::array::varbin::{varbin_scalar, VarBinArray};
use crate::array::VarBinEncoding;
use crate::compute::{CastFn, CompareFn, FilterFn, MaskFn, ScalarAtFn, SliceFn, TakeFn, ToArrowFn};
use crate::vtable::ComputeVTable;
use crate::Array;

mod cast;
mod compare;
mod filter;
mod mask;
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

    fn binary_boolean_fn(&self) -> Option<&dyn crate::compute::BinaryBooleanFn<Array>> {
        None
    }

    fn binary_numeric_fn(&self) -> Option<&dyn crate::compute::BinaryNumericFn<Array>> {
        None
    }

    fn fill_forward_fn(&self) -> Option<&dyn crate::compute::FillForwardFn<Array>> {
        None
    }

    fn fill_null_fn(&self) -> Option<&dyn crate::compute::FillNullFn<Array>> {
        None
    }

    fn invert_fn(&self) -> Option<&dyn crate::compute::InvertFn<Array>> {
        None
    }

    fn like_fn(&self) -> Option<&dyn crate::compute::LikeFn<Array>> {
        None
    }

    fn search_sorted_fn(&self) -> Option<&dyn crate::compute::SearchSortedFn<Array>> {
        None
    }

    fn search_sorted_usize_fn(&self) -> Option<&dyn crate::compute::SearchSortedUsizeFn<Array>> {
        None
    }
}

impl ScalarAtFn<VarBinArray> for VarBinEncoding {
    fn scalar_at(&self, array: &VarBinArray, index: usize) -> VortexResult<Scalar> {
        Ok(varbin_scalar(array.bytes_at(index)?, array.dtype()))
    }
}
