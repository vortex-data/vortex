use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{downcast_primitive, ArrayRef};
use vortex_dtype::DType;
use vortex_error::{vortex_bail, vortex_err, VortexError, VortexResult};
use vortex_scalar::Scalar;

use crate::encoding::Encoding;
use crate::{ArrayDType, ArrayData, IntoCanonical};

pub trait SumFn<Array> {
    fn sum(&self, array: &Array) -> VortexResult<Scalar>;

    fn sum_sq(&self, array: &Array) -> VortexResult<Scalar>;
}

impl<E: Encoding> SumFn<ArrayData> for E
where
    E: SumFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a ArrayData, Error = VortexError>,
{
    fn sum(&self, array: &ArrayData) -> VortexResult<Scalar> {
        let array_ref = <&E::Array>::try_from(array)?;
        let encoding = array
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .ok_or_else(|| vortex_err!("Mismatched encoding"))?;
        SumFn::sum(encoding, array_ref)
    }

    fn sum_sq(&self, array: &ArrayData) -> VortexResult<Scalar> {
        let array_ref = <&E::Array>::try_from(array)?;
        let encoding = array
            .encoding()
            .as_any()
            .downcast_ref::<E>()
            .ok_or_else(|| vortex_err!("Mismatched encoding"))?;
        SumFn::sum_sq(encoding, array_ref)
    }
}

#[allow(dead_code)]
pub fn sum(array: impl AsRef<ArrayData>) -> VortexResult<Scalar> {
    let array = array.as_ref();

    // if let Some(f) = array.encoding().sum_fn() {
    //     return f.sum(array);
    // }

    // if subtraction is not implemented for the given array type, but the array has a numeric
    // DType, we can flatten the array and apply subtraction to the flattened primitive array
    match array.dtype() {
        DType::Primitive(..) => {
            let arr = array.clone().into_canonical()?.into_arrow()?;

            let dt = arr.data_type();

            macro_rules! primitive_helper {
                ($T:ty) => {
                    Scalar::from(arrow_arith::aggregate::sum_array::<Int32Type, _>(
                        arr.as_primitive::<Int32Type>(),
                    ))
                };
            }

            Ok(downcast_primitive!(
                dt => (primitive_helper),
                _ => vortex_bail!("Expected numeric array")
            ))
        }
        _ => Err(vortex_err!(
            NotImplemented: "sum",
            array.encoding().id()
        )),
    }
}

#[allow(dead_code)]
fn sum_p(a: ArrayRef) -> Scalar {
    let pa = a.as_primitive::<Int32Type>();
    let res = arrow_arith::aggregate::sum_array::<Int32Type, _>(pa);
    Scalar::from(res)
}

#[allow(dead_code)]
pub fn sum_sq(array: impl AsRef<ArrayData>) -> VortexResult<Scalar> {
    let array = array.as_ref();

    if let Some(f) = array.encoding().sum_fn() {
        return f.sum_sq(array);
    }

    // if subtraction is not implemented for the given array type, but the array has a numeric
    // DType, we can flatten the array and apply subtraction to the flattened primitive array
    match array.dtype() {
        DType::Primitive(..) => {
            let arr = array.clone().into_canonical()?.into_arrow()?;

            let arr = arrow_arith::numeric::mul(&arr.clone(), &arr)?;

            let dt = arr.data_type();

            macro_rules! primitive_helper {
                ($T:ty) => {
                    Scalar::from(arrow_arith::aggregate::sum_array::<Int32Type, _>(
                        arr.as_primitive::<Int32Type>(),
                    ))
                };
            }

            Ok(downcast_primitive!(
                dt => (primitive_helper),
                _ => vortex_bail!("Expected numeric array")
            ))
        }
        _ => Err(vortex_err!(
            NotImplemented: "sum_sq",
            array.encoding().id()
        )),
    }
}

mod tests {
    use crate::array::PrimitiveArray;
    use crate::compute::sum::{sum, sum_sq};
    use crate::IntoArrayData;

    #[test]
    fn test_sum() {
        let elements = PrimitiveArray::from(vec![1i32, 2, 3, 4, 5]);

        let sum_ = sum(elements.clone().into_array()).unwrap();
        println!("sum {:?}", sum_);
        let sum_sq_ = sum_sq(elements.into_array()).unwrap();
        println!("sum sq {:?}", sum_sq_)
    }
}
