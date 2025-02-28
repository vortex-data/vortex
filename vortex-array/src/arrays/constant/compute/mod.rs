mod binary_numeric;
mod boolean;
mod cast;
mod compare;
mod invert;
mod search_sorted;

use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::arrays::ConstantEncoding;
use crate::arrays::constant::ConstantArray;
use crate::compute::{
    BinaryBooleanFn, BinaryNumericFn, CastFn, CompareFn, FilterFn, InvertFn, ScalarAtFn,
    SearchSortedFn, SliceFn, TakeFn, UncompressedSizeFn,
};
use crate::vtable::ComputeVTable;
use crate::{Array, ArrayRef};

impl ComputeVTable for ConstantEncoding {
    fn binary_boolean_fn(&self) -> Option<&dyn BinaryBooleanFn<&dyn Array>> {
        Some(self)
    }

    fn binary_numeric_fn(&self) -> Option<&dyn BinaryNumericFn<&dyn Array>> {
        Some(self)
    }

    fn cast_fn(&self) -> Option<&dyn CastFn<&dyn Array>> {
        Some(self)
    }

    fn compare_fn(&self) -> Option<&dyn CompareFn<&dyn Array>> {
        Some(self)
    }

    fn filter_fn(&self) -> Option<&dyn FilterFn<&dyn Array>> {
        Some(self)
    }

    fn invert_fn(&self) -> Option<&dyn InvertFn<&dyn Array>> {
        Some(self)
    }

    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<&dyn Array>> {
        Some(self)
    }

    fn search_sorted_fn(&self) -> Option<&dyn SearchSortedFn<&dyn Array>> {
        Some(self)
    }

    fn slice_fn(&self) -> Option<&dyn SliceFn<&dyn Array>> {
        Some(self)
    }

    fn take_fn(&self) -> Option<&dyn TakeFn<&dyn Array>> {
        Some(self)
    }

    fn uncompressed_size_fn(&self) -> Option<&dyn UncompressedSizeFn<&dyn Array>> {
        Some(self)
    }
}

impl ScalarAtFn<&ConstantArray> for ConstantEncoding {
    fn scalar_at(&self, array: &ConstantArray, _index: usize) -> VortexResult<Scalar> {
        Ok(array.scalar().clone())
    }
}

impl TakeFn<&ConstantArray> for ConstantEncoding {
    fn take(&self, array: &ConstantArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        Ok(ConstantArray::new(array.scalar().clone(), indices.len()).into_array())
    }
}

impl SliceFn<&ConstantArray> for ConstantEncoding {
    fn slice(&self, array: &ConstantArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        Ok(ConstantArray::new(array.scalar().clone(), stop - start).into_array())
    }
}

impl FilterFn<&ConstantArray> for ConstantEncoding {
    fn filter(&self, array: &ConstantArray, mask: &Mask) -> VortexResult<ArrayRef> {
        Ok(ConstantArray::new(array.scalar().clone(), mask.true_count()).into_array())
    }
}

impl UncompressedSizeFn<&ConstantArray> for ConstantEncoding {
    fn uncompressed_size(&self, array: &ConstantArray) -> VortexResult<usize> {
        let scalar = array.scalar();

        let size = match scalar.as_bool_opt() {
            Some(_) => array.len() / 8,
            None => array.scalar().nbytes() * array.len(),
        };
        Ok(size)
    }
}

#[cfg(test)]
mod test {
    use vortex_dtype::half::f16;
    use vortex_scalar::Scalar;

    use super::ConstantArray;
    use crate::array::Array;
    use crate::compute::test_harness::test_mask;

    #[test]
    fn test_mask_constant() {
        test_mask(&ConstantArray::new(Scalar::null_typed::<i32>(), 5).into_array());
        test_mask(&ConstantArray::new(Scalar::from(3u16), 5).into_array());
        test_mask(&ConstantArray::new(Scalar::from(1.0f32 / 0.0f32), 5).into_array());
        test_mask(&ConstantArray::new(Scalar::from(f16::from_f32(3.0f32)), 5).into_array());
    }
}
