use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::array::varbin::{varbin_scalar, VarBinArray};
use crate::array::VarBinEncoding;
use crate::compute::unary::ScalarAtFn;
use crate::compute::{ComputeVTable, FilterFn, SliceFn, TakeFn};
use crate::{ArrayDType, ArrayData};

mod filter;
mod slice;
mod take;

impl ComputeVTable for VarBinEncoding {
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
}

impl ScalarAtFn<VarBinArray> for VarBinEncoding {
    fn scalar_at(&self, array: &VarBinArray, index: usize) -> VortexResult<Scalar> {
        Ok(varbin_scalar(array.bytes_at(index)?, array.dtype()))
    }
}
