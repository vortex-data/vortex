use aggregate::sum_array;
use arrow_arith::aggregate;
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Float16Type, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type,
    UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrayRef, ArrowNativeTypeOp, ArrowNumericType};
use arrow_schema::DataType;
use vortex_dtype::DType;
use vortex_error::{vortex_err, VortexError, VortexResult};
use vortex_scalar::Scalar;

use crate::encoding::Encoding;
use crate::{ArrayDType, ArrayData, IntoCanonical};

pub trait SumFn<Array> {
    fn sum(&self, array: &Array, ends: &[u64]) -> VortexResult<ArrayData>;
}

pub trait FmaFn<Array> {
    // Applies the Fused Multiply-Add operation to the array, returning a new array of size |ends|
    // where each value is the sum of the array element between successive ends times by the values
    // at that index.

    // There will the size of ends matches to size of values.
    fn fma(&self, array: &Array, ends: &[u64], values: &ArrayData) -> VortexResult<ArrayData>;
}

impl<E: Encoding> SumFn<ArrayData> for E
where
    E: SumFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a ArrayData, Error = VortexError>,
{
    fn sum(&self, array: &ArrayData, ends: &[u64]) -> VortexResult<ArrayData> {
        let array_ref = <&E::Array>::try_from(array)?;
        let encoding = array
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .ok_or_else(|| vortex_err!("Mismatched encoding"))?;
        SumFn::sum(encoding, array_ref, ends)
    }
}

impl<E: Encoding> FmaFn<ArrayData> for E
where
    E: FmaFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a ArrayData, Error = VortexError>,
{
    fn fma(&self, array: &ArrayData, ends: &[u64], values: ArrayData) -> VortexResult<ArrayData> {
        let array_ref = <&E::Array>::try_from(array)?;
        let encoding = array
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .ok_or_else(|| vortex_err!("Mismatched encoding"))?;
        FmaFn::fma(encoding, array_ref, ends, values)
    }
}

#[allow(dead_code)]
pub fn sum(array: impl AsRef<ArrayData>, ends: &[u64]) -> VortexResult<ArrayData> {
    let array = array.as_ref();

    if let Some(f) = array.encoding().sum_fn() {
        return f.sum(array, ends);
    }

    let dt = array.dtype();

    // if subtraction is not implemented for the given array type, but the array has a numeric
    // DType, we can flatten the array and apply subtraction to the flattened primitive array
    match dt {
        DType::Primitive(..) => {
            let arr = array.clone().into_canonical()?.into_arrow()?;
            match arr.data_type() {
                DataType::Int8 => Ok(ends_add::<Int8Type>(arr, ends).into()),
                DataType::Int16 => Ok(ends_add::<Int16Type>(arr, ends).into()),
                DataType::Int32 => Ok(ends_add::<Int32Type>(arr, ends).into()),
                DataType::Int64 => Ok(ends_add::<Int64Type>(arr, ends).into()),
                DataType::UInt16 => Ok(ends_add::<UInt16Type>(arr, ends).into()),
                DataType::UInt32 => Ok(ends_add::<UInt32Type>(arr, ends).into()),
                DataType::UInt64 => Ok(ends_add::<UInt64Type>(arr, ends).into()),
                DataType::Float16 => Ok(ends_add::<Float16Type>(arr, ends).into()),
                DataType::Float32 => Ok(ends_add::<Float32Type>(arr, ends).into()),
                DataType::Float64 => Ok(ends_add::<Float64Type>(arr, ends).into()),
                _ => todo!(),
            }
        }
        _ => Err(vortex_err!(
            NotImplemented: "sum",
            array.encoding().id()
        )),
    }
}

fn ends_add<T>(arr: ArrayRef, ends: &[u64]) -> Vec<T::Native>
where
    T: ArrowNumericType,
    T::Native: ArrowNativeTypeOp,
{
    let mut res = Vec::with_capacity(ends.len());
    let mut start = 0;
    let prim = arr.as_primitive::<T>();
    for &end in ends {
        let slice = prim.slice(start, end as usize);
        res.push(
            sum_array::<T, _>(&slice)
                .map(|v| v)
                .unwrap_or(T::Native::ZERO),
        );
        start = end as usize;
    }
    res
}

#[allow(dead_code)]
fn sum_p(a: ArrayRef) -> Scalar {
    let pa = a.as_primitive::<Int32Type>();
    let res = sum_array::<Int32Type, _>(pa);
    Scalar::from(res)
}

#[cfg(test)]
mod tests {
    use crate::array::PrimitiveArray;
    use crate::compute::sum::sum;
    use crate::{IntoArrayData, IntoArrayVariant};

    #[test]
    fn test_sum() {
        let elements = PrimitiveArray::from(vec![1i32, 2, 3, 4, 5]);

        let sum_ = sum(elements.clone().into_array(), vec![5].as_slice()).unwrap();
        assert_eq!(
            sum_.into_primitive().unwrap().maybe_null_slice::<i32>(),
            &[15]
        );
    }
}
