mod boolean;
mod compare;
mod search_sorted;

use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::array::constant::ConstantArray;
use crate::array::ConstantEncoding;
use crate::compute::unary::ScalarAtFn;
use crate::compute::{
    BinaryBooleanFn, CompareFn, ComputeVTable, FilterFn, FilterMask, SearchSortedFn, SliceFn,
    TakeFn, TakeOptions,
};
use crate::{ArrayData, IntoArrayData};

impl ComputeVTable for ConstantEncoding {
    fn binary_boolean_fn(
        &self,
        lhs: &ArrayData,
        rhs: &ArrayData,
    ) -> Option<&dyn BinaryBooleanFn<ArrayData>> {
        // We only need to deal with this if both sides are constant, otherwise other arrays
        // will have handled the RHS being constant.
        (lhs.is_constant() && rhs.is_constant()).then_some(self)
    }

    fn compare_fn(&self) -> Option<&dyn CompareFn<ArrayData>> {
        Some(self)
    }

    fn filter_fn(&self) -> Option<&dyn FilterFn<ArrayData>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<ArrayData>> {
        Some(self)
    }

    fn search_sorted_fn(&self) -> Option<&dyn SearchSortedFn<ArrayData>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<ArrayData>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<ArrayData>> {
        Some(self)
    }
}

impl ScalarAtFn<ConstantArray> for ConstantEncoding {
    fn scalar_at(&self, array: &ConstantArray, _index: usize) -> VortexResult<Scalar> {
        Ok(array.scalar())
    }
}

impl TakeFn<ConstantArray> for ConstantEncoding {
    fn take(
        &self,
        array: &ConstantArray,
        indices: &ArrayData,
        _options: TakeOptions,
    ) -> VortexResult<ArrayData> {
        Ok(ConstantArray::new(array.scalar(), indices.len()).into_array())
    }
}

impl SliceFn<ConstantArray> for ConstantEncoding {
    fn slice(&self, array: &ConstantArray, start: usize, stop: usize) -> VortexResult<ArrayData> {
        Ok(ConstantArray::new(array.scalar(), stop - start).into_array())
    }
}

impl FilterFn<ConstantArray> for ConstantEncoding {
    fn filter(&self, array: &ConstantArray, mask: FilterMask) -> VortexResult<ArrayData> {
        Ok(ConstantArray::new(array.scalar(), mask.true_count()).into_array())
    }
}
