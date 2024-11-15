use vortex_error::{VortexResult, VortexUnwrap as _};
use vortex_scalar::Scalar;

use crate::array::varbin::{varbin_scalar, VarBinArray};
use crate::compute::unary::ScalarAtFn;
use crate::compute::{ArrayCompute, FilterFn, MaybeCompareFn, Operator, SliceFn, TakeFn};
use crate::{ArrayDType, ArrayData};

mod compare;
mod filter;
mod slice;
mod take;

impl ArrayCompute for VarBinArray {
    fn compare(&self, other: &ArrayData, operator: Operator) -> Option<VortexResult<ArrayData>> {
        MaybeCompareFn::maybe_compare(self, other, operator)
    }

    fn filter(&self) -> Option<&dyn FilterFn> {
        Some(self)
    }

    fn scalar_at(&self) -> Option<&dyn ScalarAtFn> {
        Some(self)
    }

    fn slice(&self) -> Option<&dyn SliceFn> {
        Some(self)
    }

    fn take(&self) -> Option<&dyn TakeFn> {
        Some(self)
    }
}

impl ScalarAtFn for VarBinArray {
    fn scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        Ok(varbin_scalar(self.bytes_at(index)?, self.dtype()))
    }

    fn scalar_at_unchecked(&self, index: usize) -> Scalar {
        varbin_scalar(self.bytes_at(index).vortex_unwrap(), self.dtype())
    }
}
