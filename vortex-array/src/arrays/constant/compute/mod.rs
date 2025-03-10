mod binary_numeric;
mod boolean;
mod cast;
mod compare;
mod invert;
mod search_sorted;

use num_traits::{CheckedMul, ToPrimitive};
use vortex_dtype::{NativePType, PType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_err};
use vortex_mask::Mask;
use vortex_scalar::{FromPrimitiveOrF16, PrimitiveScalar, Scalar};

use crate::arrays::ConstantEncoding;
use crate::arrays::constant::ConstantArray;
use crate::compute::{
    BinaryBooleanFn, BinaryNumericFn, CastFn, CompareFn, FilterFn, InvertFn, ScalarAtFn,
    SearchSortedFn, SliceFn, SumFn, TakeFn, UncompressedSizeFn,
};
use crate::stats::Stat;
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

    fn sum_fn(&self) -> Option<&dyn SumFn<&dyn Array>> {
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

impl SumFn<&ConstantArray> for ConstantEncoding {
    fn sum(&self, array: &ConstantArray) -> VortexResult<Scalar> {
        let sum_dtype = Stat::Sum
            .dtype(array.dtype())
            .ok_or_else(|| vortex_err!("Sum not supported for dtype {}", array.dtype()))?;
        let sum_ptype = PType::try_from(&sum_dtype).vortex_expect("sum dtype must be primitive");

        let scalar = array.scalar();

        let scalar_value = match_each_native_ptype!(
            sum_ptype,
            unsigned: |$T| { sum_integral::<u64>(scalar.as_primitive(), array.len())?.into() }
            signed: |$T| { sum_integral::<i64>(scalar.as_primitive(), array.len())?.into() }
            floating: |$T| { sum_float(scalar.as_primitive(), array.len())?.into() }
        );

        Ok(Scalar::new(sum_dtype, scalar_value))
    }
}

fn sum_integral<T>(
    primitive_scalar: PrimitiveScalar<'_>,
    array_len: usize,
) -> VortexResult<Option<T>>
where
    T: FromPrimitiveOrF16 + NativePType + CheckedMul,
    Scalar: From<Option<T>>,
{
    let v = primitive_scalar.as_::<T>()?;
    let array_len =
        T::from(array_len).ok_or_else(|| vortex_err!("array_len must fit the sum type"))?;
    let sum = v.and_then(|v| v.checked_mul(&array_len));

    Ok(sum)
}

fn sum_float(primitive_scalar: PrimitiveScalar<'_>, array_len: usize) -> VortexResult<Option<f64>> {
    let v = primitive_scalar.as_::<f64>()?;
    let array_len = array_len
        .to_f64()
        .ok_or_else(|| vortex_err!("array_len must fit the sum type"))?;

    Ok(v.map(|v| v * array_len))
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
