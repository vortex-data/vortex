use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::array::varbin::{varbin_scalar, VarBinArray};
use crate::array::VarBinEncoding;
use crate::compute::{CompareFn, FilterFn, ScalarAtFn, SliceFn, TakeFn, ToArrowFn};
use crate::vtable::ComputeVTable;
use crate::{ArrayDType, ArrayData};

mod compare;
mod filter;
mod slice;
mod take;
pub(crate) mod to_arrow;

impl ComputeVTable for VarBinEncoding {
    fn compare_fn(&self) -> Option<&dyn CompareFn<ArrayData>> {
        Some(self)
    }

    fn filter_fn(&self) -> Option<&dyn FilterFn<ArrayData>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<ArrayData>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<ArrayData>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<ArrayData>> {
        Some(self)
    }

    fn to_arrow_fn(&self) -> Option<&dyn ToArrowFn<ArrayData>> {
        Some(self)
    }
}

impl ScalarAtFn<VarBinArray> for VarBinEncoding {
    fn scalar_at(&self, array: &VarBinArray, index: usize) -> VortexResult<Scalar> {
        Ok(varbin_scalar(array.bytes_at(index)?, array.dtype()))
    }
}
