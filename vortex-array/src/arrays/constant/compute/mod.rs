mod binary_numeric;
mod boolean;
mod cast;
mod compare;
mod invert;
mod search_sorted;

use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::arrays::constant::ConstantArray;
use crate::arrays::ConstantEncoding;
use crate::compute::{
    BinaryBooleanFn, BinaryNumericFn, CastFn, CompareFn, FilterFn, InvertFn, ScalarAtFn,
    SearchSortedFn, SliceFn, TakeFn,
};
use crate::vtable::ComputeVTable;
use crate::{Array, IntoArray};

impl ComputeVTable for ConstantEncoding {
    fn binary_boolean_fn(&self) -> Option<&dyn BinaryBooleanFn<Array>> {
        Some(self)
    }

    fn binary_numeric_fn(&self) -> Option<&dyn BinaryNumericFn<Array>> {
        Some(self)
    }

    fn cast_fn(&self) -> Option<&dyn CastFn<Array>> {
        Some(self)
    }

    fn compare_fn(&self) -> Option<&dyn CompareFn<Array>> {
        Some(self)
    }

    fn filter_fn(&self) -> Option<&dyn FilterFn<Array>> {
        Some(self)
    }

    fn invert_fn(&self) -> Option<&dyn InvertFn<Array>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<Array>> {
        Some(self)
    }

    fn search_sorted_fn(&self) -> Option<&dyn SearchSortedFn<Array>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<Array>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<Array>> {
        Some(self)
    }
}

impl ScalarAtFn<ConstantArray> for ConstantEncoding {
    fn scalar_at(&self, array: &ConstantArray, _index: usize) -> VortexResult<Scalar> {
        Ok(array.scalar())
    }
}

impl TakeFn<ConstantArray> for ConstantEncoding {
    fn take(&self, array: &ConstantArray, indices: &Array) -> VortexResult<Array> {
        Ok(ConstantArray::new(array.scalar(), indices.len()).into_array())
    }
}

impl SliceFn<ConstantArray> for ConstantEncoding {
    fn slice(&self, array: &ConstantArray, start: usize, stop: usize) -> VortexResult<Array> {
        Ok(ConstantArray::new(array.scalar(), stop - start).into_array())
    }
}

impl FilterFn<ConstantArray> for ConstantEncoding {
    fn filter(&self, array: &ConstantArray, mask: &Mask) -> VortexResult<Array> {
        Ok(ConstantArray::new(array.scalar(), mask.true_count()).into_array())
    }
}

#[cfg(test)]
mod test {
    use vortex_dtype::half::f16;
    use vortex_scalar::Scalar;

    use super::ConstantArray;
    use crate::compute::test_harness::test_mask;
    use crate::IntoArray as _;

    #[test]
    fn test_mask_constant() {
        test_mask(ConstantArray::new(Scalar::null_typed::<i32>(), 5).into_array());
        test_mask(ConstantArray::new(Scalar::from(3u16), 5).into_array());
        test_mask(ConstantArray::new(Scalar::from(1.0f32 / 0.0f32), 5).into_array());
        test_mask(ConstantArray::new(Scalar::from(f16::from_f32(3.0f32)), 5).into_array());
    }
}
