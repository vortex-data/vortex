mod boolean;
mod compare;

use std::cmp::Ordering;

use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::array::constant::ConstantArray;
use crate::array::ConstantEncoding;
use crate::compute::unary::ScalarAtFn;
use crate::compute::{
    ArrayCompute, BinaryBooleanFn, CompareFn, ComputeVTable, FilterFn, FilterMask, SearchResult,
    SearchSortedFn, SearchSortedSide, SliceFn, TakeFn, TakeOptions,
};
use crate::{ArrayData, ArrayLen, IntoArrayData};

impl ArrayCompute for ConstantArray {
    fn compare(&self) -> Option<&dyn CompareFn> {
        Some(self)
    }

    fn search_sorted(&self) -> Option<&dyn SearchSortedFn> {
        Some(self)
    }
}

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

impl ScalarAtFn<ConstantArray> for ConstantEncoding {
    fn scalar_at(&self, array: &ConstantArray, _index: usize) -> VortexResult<Scalar> {
        Ok(array.owned_scalar())
    }
}

impl TakeFn<ConstantArray> for ConstantEncoding {
    fn take(
        &self,
        array: &ConstantArray,
        indices: &ArrayData,
        _options: TakeOptions,
    ) -> VortexResult<ArrayData> {
        Ok(ConstantArray::new(array.owned_scalar(), indices.len()).into_array())
    }
}

impl SliceFn<ConstantArray> for ConstantEncoding {
    fn slice(&self, array: &ConstantArray, start: usize, stop: usize) -> VortexResult<ArrayData> {
        Ok(ConstantArray::new(array.owned_scalar(), stop - start).into_array())
    }
}

impl FilterFn<ConstantArray> for ConstantEncoding {
    fn filter(&self, array: &ConstantArray, mask: FilterMask) -> VortexResult<ArrayData> {
        Ok(ConstantArray::new(array.owned_scalar(), mask.true_count()).into_array())
    }
}

impl SearchSortedFn for ConstantArray {
    fn search_sorted(&self, value: &Scalar, side: SearchSortedSide) -> VortexResult<SearchResult> {
        match self
            .scalar_value()
            .partial_cmp(value.value())
            .unwrap_or(Ordering::Less)
        {
            Ordering::Greater => Ok(SearchResult::NotFound(0)),
            Ordering::Less => Ok(SearchResult::NotFound(self.len())),
            Ordering::Equal => match side {
                SearchSortedSide::Left => Ok(SearchResult::Found(0)),
                SearchSortedSide::Right => Ok(SearchResult::Found(self.len())),
            },
        }
    }
}

#[cfg(test)]
mod test {
    use crate::array::constant::ConstantArray;
    use crate::compute::{search_sorted, SearchResult, SearchSortedSide};
    use crate::IntoArrayData;

    #[test]
    pub fn search() {
        let cst = ConstantArray::new(42, 5000).into_array();
        assert_eq!(
            search_sorted(&cst, 33, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(0)
        );
        assert_eq!(
            search_sorted(&cst, 55, SearchSortedSide::Left).unwrap(),
            SearchResult::NotFound(5000)
        );
    }

    #[test]
    pub fn search_equals() {
        let cst = ConstantArray::new(42, 5000).into_array();
        assert_eq!(
            search_sorted(&cst, 42, SearchSortedSide::Left).unwrap(),
            SearchResult::Found(0)
        );
        assert_eq!(
            search_sorted(&cst, 42, SearchSortedSide::Right).unwrap(),
            SearchResult::Found(5000)
        );
    }
}
