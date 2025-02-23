use vortex_dtype::{DType, PType};
use vortex_error::{vortex_bail, VortexExpect, VortexResult};
use vortex_scalar::{BinaryNumericOperator, Scalar};

use crate::arrays::ConstantArray;
use crate::arrow::{from_arrow_array_with_len, Datum};
use crate::encoding::Encoding;
use crate::{Array, ArrayRef};

pub trait BinaryNumericFn<A> {
    fn binary_numeric(
        &self,
        array: A,
        other: &dyn Array,
        op: BinaryNumericOperator,
    ) -> VortexResult<Option<ArrayRef>>;
}

impl<E: Encoding> BinaryNumericFn<&dyn Array> for E
where
    E: for<'a> BinaryNumericFn<&'a E::Array>,
{
    fn binary_numeric(
        &self,
        lhs: &dyn Array,
        rhs: &dyn Array,
        op: BinaryNumericOperator,
    ) -> VortexResult<Option<ArrayRef>> {
        let array_ref = lhs
            .as_any()
            .downcast_ref::<E::Array>()
            .vortex_expect("Failed to downcast array");
        BinaryNumericFn::binary_numeric(self, array_ref, rhs, op)
    }
}

/// Point-wise add two numeric arrays.
pub fn add(lhs: &dyn Array, rhs: &dyn Array) -> VortexResult<ArrayRef> {
    binary_numeric(lhs, rhs, BinaryNumericOperator::Add)
}

/// Point-wise add a scalar value to this array on the right-hand-side.
pub fn add_scalar(lhs: &dyn Array, rhs: Scalar) -> VortexResult<ArrayRef> {
    binary_numeric(
        lhs,
        &ConstantArray::new(rhs, lhs.len()).into_array(),
        BinaryNumericOperator::Add,
    )
}

/// Point-wise subtract two numeric arrays.
pub fn sub(lhs: &dyn Array, rhs: &dyn Array) -> VortexResult<ArrayRef> {
    binary_numeric(lhs, rhs, BinaryNumericOperator::Sub)
}

/// Point-wise subtract a scalar value from this array on the right-hand-side.
pub fn sub_scalar(lhs: &dyn Array, rhs: Scalar) -> VortexResult<ArrayRef> {
    binary_numeric(
        lhs,
        &ConstantArray::new(rhs, lhs.len()).into_array(),
        BinaryNumericOperator::Sub,
    )
}

/// Point-wise multiply two numeric arrays.
pub fn mul(lhs: &dyn Array, rhs: &dyn Array) -> VortexResult<ArrayRef> {
    binary_numeric(lhs, rhs, BinaryNumericOperator::Mul)
}

/// Point-wise multiply a scalar value into this array on the right-hand-side.
pub fn mul_scalar(lhs: &dyn Array, rhs: Scalar) -> VortexResult<ArrayRef> {
    binary_numeric(
        lhs,
        &ConstantArray::new(rhs, lhs.len()).into_array(),
        BinaryNumericOperator::Mul,
    )
}

/// Point-wise divide two numeric arrays.
pub fn div(lhs: &dyn Array, rhs: &dyn Array) -> VortexResult<ArrayRef> {
    binary_numeric(lhs, rhs, BinaryNumericOperator::Div)
}

/// Point-wise divide a scalar value into this array on the right-hand-side.
pub fn div_scalar(lhs: &dyn Array, rhs: Scalar) -> VortexResult<ArrayRef> {
    binary_numeric(
        lhs,
        &ConstantArray::new(rhs, lhs.len()).into_array(),
        BinaryNumericOperator::Mul,
    )
}

pub fn binary_numeric(
    lhs: &dyn Array,
    rhs: &dyn Array,
    op: BinaryNumericOperator,
) -> VortexResult<ArrayRef> {
    if lhs.len() != rhs.len() {
        vortex_bail!(
            "Numeric operations aren't supported on arrays of different lengths {} {}",
            lhs.len(),
            rhs.len()
        )
    }
    if !matches!(lhs.dtype(), DType::Primitive(_, _))
        || !matches!(rhs.dtype(), DType::Primitive(_, _))
        || lhs.dtype() != rhs.dtype()
    {
        vortex_bail!(
            "Numeric operations are only supported on two arrays sharing the same primitive-type: {} {}",
            lhs.dtype(),
            rhs.dtype()
        )
    }

    // Check if LHS supports the operation directly.
    if let Some(fun) = lhs.vtable().binary_numeric_fn() {
        if let Some(result) = fun.binary_numeric(lhs, rhs, op)? {
            check_numeric_result(&result, lhs, rhs);
            return Ok(result);
        }
    }

    // Check if RHS supports the operation directly.
    if let Some(fun) = rhs.vtable().binary_numeric_fn() {
        if let Some(result) = fun.binary_numeric(rhs, lhs, op.swap())? {
            check_numeric_result(&result, lhs, rhs);
            return Ok(result);
        }
    }

    log::debug!(
        "No numeric implementation found for LHS {}, RHS {}, and operator {:?}",
        lhs.encoding(),
        rhs.encoding(),
        op,
    );

    // If neither side implements the trait, then we delegate to Arrow compute.
    arrow_numeric(lhs, rhs, op)
}

/// Implementation of `BinaryBooleanFn` using the Arrow crate.
///
/// Note that other encodings should handle a constant RHS value, so we can assume here that
/// the RHS is not constant and expand to a full array.
fn arrow_numeric(
    lhs: &dyn Array,
    rhs: &dyn Array,
    operator: BinaryNumericOperator,
) -> VortexResult<ArrayRef> {
    let nullable = lhs.dtype().is_nullable() || rhs.dtype().is_nullable();
    let len = lhs.len();

    let left = Datum::try_new(lhs.to_array())?;
    let right = Datum::try_new(rhs.to_array())?;

    let array = match operator {
        BinaryNumericOperator::Add => arrow_arith::numeric::add(&left, &right)?,
        BinaryNumericOperator::Sub => arrow_arith::numeric::sub(&left, &right)?,
        BinaryNumericOperator::RSub => arrow_arith::numeric::sub(&right, &left)?,
        BinaryNumericOperator::Mul => arrow_arith::numeric::mul(&left, &right)?,
        BinaryNumericOperator::Div => arrow_arith::numeric::div(&left, &right)?,
        BinaryNumericOperator::RDiv => arrow_arith::numeric::div(&right, &left)?,
    };

    let result = from_arrow_array_with_len(array, len, nullable)?;
    check_numeric_result(&result, lhs, rhs);
    Ok(result)
}

#[inline(always)]
fn check_numeric_result(result: &dyn Array, lhs: &dyn Array, rhs: &dyn Array) {
    debug_assert_eq!(
        result.len(),
        lhs.len(),
        "Numeric operation length mismatch {}",
        rhs.encoding()
    );
    debug_assert_eq!(
        result.dtype(),
        &DType::Primitive(
            PType::try_from(lhs.dtype())
                .vortex_expect("Numeric operation DType failed to convert to PType"),
            (lhs.dtype().is_nullable() || rhs.dtype().is_nullable()).into()
        ),
        "Numeric operation dtype mismatch {}",
        rhs.encoding()
    );
}

#[cfg(feature = "test-harness")]
pub mod test_harness {
    use num_traits::Num;
    use vortex_dtype::NativePType;
    use vortex_error::{vortex_err, VortexResult};
    use vortex_scalar::{BinaryNumericOperator, PrimitiveScalar, Scalar};

    use crate::arrays::ConstantArray;
    use crate::compute::{binary_numeric, scalar_at};
    use crate::{Array, ArrayRef};

    #[allow(clippy::unwrap_used)]
    fn to_vec_of_scalar(array: &dyn Array) -> Vec<Scalar> {
        // Not fast, but obviously correct
        (0..array.len())
            .map(|index| scalar_at(array, index))
            .collect::<VortexResult<Vec<_>>>()
            .unwrap()
    }

    #[allow(clippy::unwrap_used)]
    pub fn test_binary_numeric<T: NativePType + Num + Copy>(array: ArrayRef)
    where
        Scalar: From<T>,
    {
        let canonicalized_array = array.to_canonical().unwrap().into_primitive().unwrap();
        let original_values = to_vec_of_scalar(&canonicalized_array.into_array());

        let one = T::from(1)
            .ok_or_else(|| vortex_err!("could not convert 1 into array native type"))
            .unwrap();
        let scalar_one = Scalar::from(one).cast(array.dtype()).unwrap();

        let operators: [BinaryNumericOperator; 6] = [
            BinaryNumericOperator::Add,
            BinaryNumericOperator::Sub,
            BinaryNumericOperator::RSub,
            BinaryNumericOperator::Mul,
            BinaryNumericOperator::Div,
            BinaryNumericOperator::RDiv,
        ];

        for operator in operators {
            assert_eq!(
                to_vec_of_scalar(
                    &binary_numeric(
                        &array,
                        &ConstantArray::new(scalar_one.clone(), array.len()).into_array(),
                        operator
                    )
                    .unwrap()
                ),
                original_values
                    .iter()
                    .map(|x| x
                        .as_primitive()
                        .checked_binary_numeric(scalar_one.as_primitive(), operator)
                        .unwrap()
                        .unwrap())
                    .map(<Scalar as From<PrimitiveScalar<'_>>>::from)
                    .collect::<Vec<Scalar>>(),
                "({:?}) {} (Constant array of {}) did not produce expected results",
                array,
                operator,
                scalar_one,
            );

            assert_eq!(
                to_vec_of_scalar(
                    &binary_numeric(
                        &ConstantArray::new(scalar_one.clone(), array.len()).into_array(),
                        &array,
                        operator
                    )
                    .unwrap()
                ),
                original_values
                    .iter()
                    .map(|x| scalar_one
                        .as_primitive()
                        .checked_binary_numeric(x.as_primitive(), operator)
                        .unwrap()
                        .unwrap())
                    .map(<Scalar as From<PrimitiveScalar<'_>>>::from)
                    .collect::<Vec<_>>(),
                "(Constant array of {}) {} ({:?}) did not produce expected results",
                scalar_one,
                operator,
                array,
            );
        }
    }
}

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_scalar::Scalar;

    use crate::array::Array;
    use crate::arrays::PrimitiveArray;
    use crate::canonical::ToCanonical;
    use crate::compute::{scalar_at, sub_scalar};
    use crate::IntoArray;

    #[test]
    fn test_scalar_subtract_unsigned() {
        let values = buffer![1u16, 2, 3].into_array();
        let results = sub_scalar(&values, 1u16.into())
            .unwrap()
            .to_primitive()
            .unwrap()
            .as_slice::<u16>()
            .to_vec();
        assert_eq!(results, &[0u16, 1, 2]);
    }

    #[test]
    fn test_scalar_subtract_signed() {
        let values = buffer![1i64, 2, 3].into_array();
        let results = sub_scalar(&values, (-1i64).into())
            .unwrap()
            .to_primitive()
            .unwrap()
            .as_slice::<i64>()
            .to_vec();
        assert_eq!(results, &[2i64, 3, 4]);
    }

    #[test]
    fn test_scalar_subtract_nullable() {
        let values = PrimitiveArray::from_option_iter([Some(1u16), Some(2), None, Some(3)]);
        let result = sub_scalar(&values, Some(1u16).into())
            .unwrap()
            .to_primitive()
            .unwrap();

        let actual = (0..result.len())
            .map(|index| scalar_at(&result, index).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            vec![
                Scalar::from(Some(0u16)),
                Scalar::from(Some(1u16)),
                Scalar::from(None::<u16>),
                Scalar::from(Some(2u16))
            ]
        );
    }

    #[test]
    fn test_scalar_subtract_float() {
        let values = buffer![1.0f64, 2.0, 3.0].into_array();
        let to_subtract = -1f64;
        let results = sub_scalar(&values, to_subtract.into())
            .unwrap()
            .to_primitive()
            .unwrap()
            .as_slice::<f64>()
            .to_vec();
        assert_eq!(results, &[2.0f64, 3.0, 4.0]);
    }

    #[test]
    fn test_scalar_subtract_float_underflow_is_ok() {
        let values = buffer![f32::MIN, 2.0, 3.0].into_array();
        let _results = sub_scalar(&values, 1.0f32.into()).unwrap();
        let _results = sub_scalar(&values, f32::MAX.into()).unwrap();
    }

    #[test]
    fn test_scalar_subtract_type_mismatch_fails() {
        let values = buffer![1u64, 2, 3].into_array();
        // Subtracting incompatible dtypes should fail
        let _results =
            sub_scalar(&values, 1.5f64.into()).expect_err("Expected type mismatch error");
    }
}
