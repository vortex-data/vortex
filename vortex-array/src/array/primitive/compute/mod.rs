use vortex_error::VortexResult;

use crate::array::primitive::PrimitiveArray;
use crate::compute::unary::{CastFn, FillForwardFn, ScalarAtFn, SubtractScalarFn};
use crate::compute::{ArrayCompute, MaybeCompareFn, Operator, SearchSortedFn, SliceFn, TakeFn};
use crate::ArrayData;

mod cast;
mod compare;
mod fill;
mod filter;
mod scalar_at;
mod search_sorted;
mod slice;
mod subtract_scalar;
mod take;

impl ArrayCompute for PrimitiveArray {
    fn cast(&self) -> Option<&dyn CastFn> {
        Some(self)
    }

    fn compare(&self, other: &ArrayData, operator: Operator) -> Option<VortexResult<ArrayData>> {
        MaybeCompareFn::maybe_compare(self, other, operator)
    }

    fn fill_forward(&self) -> Option<&dyn FillForwardFn> {
        Some(self)
    }

    fn scalar_at(&self) -> Option<&dyn ScalarAtFn> {
        Some(self)
    }

    fn subtract_scalar(&self) -> Option<&dyn SubtractScalarFn> {
        Some(self)
    }

    fn search_sorted(&self) -> Option<&dyn SearchSortedFn> {
        Some(self)
    }

    fn slice(&self) -> Option<&dyn SliceFn> {
        Some(self)
    }

    fn take(&self) -> Option<&dyn TakeFn> {
        Some(self)
    }
}
