mod binary_numeric;
mod boolean;
mod cast;
mod compare;
mod filter;
mod invert;
mod search_sorted;
mod sum;
mod take;

use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::Array;
use crate::arrays::ConstantEncoding;
use crate::arrays::constant::ConstantArray;
use crate::compute::{ScalarAtFn, SearchSortedFn, TakeFn, UncompressedSizeFn};
use crate::vtable::ComputeVTable;

impl ComputeVTable for ConstantEncoding {
    fn scalar_at_fn(&self) -> Option<&dyn ScalarAtFn<&dyn Array>> {
        Some(self)
    }

    fn search_sorted_fn(&self) -> Option<&dyn SearchSortedFn<&dyn Array>> {
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
    use crate::compute::conformance::mask::test_mask;

    #[test]
    fn test_mask_constant() {
        test_mask(&ConstantArray::new(Scalar::null_typed::<i32>(), 5).into_array());
        test_mask(&ConstantArray::new(Scalar::from(3u16), 5).into_array());
        test_mask(&ConstantArray::new(Scalar::from(1.0f32 / 0.0f32), 5).into_array());
        test_mask(&ConstantArray::new(Scalar::from(f16::from_f32(3.0f32)), 5).into_array());
    }
}
