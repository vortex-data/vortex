use vortex_dtype::{DType, PType};
use vortex_error::{vortex_bail, VortexError, VortexExpect, VortexResult};
use vortex_scalar::{BinaryNumericOperator, Scalar};

use crate::arrays::ConstantArray;
use crate::arrow::{from_arrow_array_with_len, Datum};
use crate::encoding::Encoding;
use crate::{Array, IntoArray as _};

pub trait BinaryNumericFn<A> {
    fn binary_numeric(
        &self,
        array: &A,
        other: &Array,
        op: BinaryNumericOperator,
    ) -> VortexResult<Option<Array>>;
}

impl<E: Encoding> BinaryNumericFn<Array> for E
where
    E: BinaryNumericFn<E::Array>,
    for<'a> &'a E::Array: TryFrom<&'a Array, Error = VortexError>,
{
    fn binary_numeric(
        &self,
        lhs: &Array,
        rhs: &Array,
        op: BinaryNumericOperator,
    ) -> VortexResult<Option<Array>> {
        let (array_ref, encoding) = lhs.try_downcast_ref::<E>()?;
        BinaryNumericFn::binary_numeric(encoding, array_ref, rhs, op)
    }
}

/// Point-wise add two numeric arrays.
pub fn add(lhs: impl AsRef<Array>, rhs: impl AsRef<Array>) -> VortexResult<Array> {
    binary_numeric(lhs.as_ref(), rhs.as_ref(), BinaryNumericOperator::Add)
}

/// Point-wise add a scalar value to this array on the right-hand-side.
pub fn add_scalar(lhs: impl AsRef<Array>, rhs: Scalar) -> VortexResult<Array> {
    let lhs = lhs.as_ref();
    binary_numeric(
        lhs,
        &ConstantArray::new(rhs, lhs.len()).into_array(),
        BinaryNumericOperator::Add,
    )
}

/// Point-wise subtract two numeric arrays.
pub fn sub(lhs: impl AsRef<Array>, rhs: impl AsRef<Array>) -> VortexResult<Array> {
    binary_numeric(lhs.as_ref(), rhs.as_ref(), BinaryNumericOperator::Sub)
}

/// Point-wise subtract a scalar value from this array on the right-hand-side.
pub fn sub_scalar(lhs: impl AsRef<Array>, rhs: Scalar) -> VortexResult<Array> {
    let lhs = lhs.as_ref();
    binary_numeric(
        lhs,
        &ConstantArray::new(rhs, lhs.len()).into_array(),
        BinaryNumericOperator::Sub,
    )
}

/// Point-wise multiply two numeric arrays.
pub fn mul(lhs: impl AsRef<Array>, rhs: impl AsRef<Array>) -> VortexResult<Array> {
    binary_numeric(lhs.as_ref(), rhs.as_ref(), BinaryNumericOperator::Mul)
}

/// Point-wise multiply a scalar value into this array on the right-hand-side.
pub fn mul_scalar(lhs: impl AsRef<Array>, rhs: Scalar) -> VortexResult<Array> {
    let lhs = lhs.as_ref();
    binary_numeric(
        lhs,
        &ConstantArray::new(rhs, lhs.len()).into_array(),
        BinaryNumericOperator::Mul,
    )
}

/// Point-wise divide two numeric arrays.
pub fn div(lhs: impl AsRef<Array>, rhs: impl AsRef<Array>) -> VortexResult<Array> {
    binary_numeric(lhs.as_ref(), rhs.as_ref(), BinaryNumericOperator::Div)
}

/// Point-wise divide a scalar value into this array on the right-hand-side.
pub fn div_scalar(lhs: impl AsRef<Array>, rhs: Scalar) -> VortexResult<Array> {
    let lhs = lhs.as_ref();
    binary_numeric(
        lhs,
        &ConstantArray::new(rhs, lhs.len()).into_array(),
        BinaryNumericOperator::Mul,
    )
}

pub fn binary_numeric(lhs: &Array, rhs: &Array, op: BinaryNumericOperator) -> VortexResult<Array> {
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
    arrow_numeric(lhs.clone(), rhs.clone(), op)
}

/// Implementation of `BinaryBooleanFn` using the Arrow crate.
///
/// Note that other encodings should handle a constant RHS value, so we can assume here that
/// the RHS is not constant and expand to a full array.
fn arrow_numeric(lhs: Array, rhs: Array, operator: BinaryNumericOperator) -> VortexResult<Array> {
    let nullable = lhs.dtype().is_nullable() || rhs.dtype().is_nullable();
    let len = lhs.len();

    let left = Datum::try_new(lhs.clone())?;
    let right = Datum::try_new(rhs.clone())?;

    let array = match operator {
        BinaryNumericOperator::Add => arrow_arith::numeric::add(&left, &right)?,
        BinaryNumericOperator::Sub => arrow_arith::numeric::sub(&left, &right)?,
        BinaryNumericOperator::RSub => arrow_arith::numeric::sub(&right, &left)?,
        BinaryNumericOperator::Mul => arrow_arith::numeric::mul(&left, &right)?,
        BinaryNumericOperator::Div => arrow_arith::numeric::div(&left, &right)?,
        BinaryNumericOperator::RDiv => arrow_arith::numeric::div(&right, &left)?,
    };

    let result = from_arrow_array_with_len(array, len, nullable)?;
    check_numeric_result(&result, &lhs, &rhs);
    Ok(result)
}

#[inline(always)]
fn check_numeric_result(result: &Array, lhs: &Array, rhs: &Array) {
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
    use crate::{Array, IntoArray as _, IntoCanonical};

    #[allow(clippy::unwrap_used)]
    fn to_vec_of_scalar(array: &Array) -> Vec<Scalar> {
        // Not fast, but obviously correct
        (0..array.len())
            .map(|index| scalar_at(array, index))
            .collect::<VortexResult<Vec<_>>>()
            .unwrap()
    }

    #[allow(clippy::unwrap_used)]
    pub fn test_binary_numeric<T: NativePType + Num + Copy>(array: Array)
    where
        Scalar: From<T>,
    {
        let canonicalized_array = array
            .clone()
            .into_canonical()
            .unwrap()
            .into_primitive()
            .unwrap();

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
                "({}) {} (Constant array of {}) did not produce expected results",
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
                "(Constant array of {}) {} ({}) did not produce expected results",
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

    use crate::arrays::PrimitiveArray;
    use crate::canonical::IntoCanonical;
    use crate::compute::{scalar_at, sub_scalar};
    use crate::IntoArray;

    #[test]
    fn test_scalar_subtract_unsigned() {
        let values = buffer![1u16, 2, 3].into_array();
        let results = sub_scalar(&values, 1u16.into())
            .unwrap()
            .into_canonical()
            .unwrap()
            .into_primitive()
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
            .into_canonical()
            .unwrap()
            .into_primitive()
            .unwrap()
            .as_slice::<i64>()
            .to_vec();
        assert_eq!(results, &[2i64, 3, 4]);
    }

    #[test]
    fn test_scalar_subtract_nullable() {
        let values =
            PrimitiveArray::from_option_iter([Some(1u16), Some(2), None, Some(3)]).into_array();
        let result = sub_scalar(&values, Some(1u16).into())
            .unwrap()
            .into_canonical()
            .unwrap()
            .into_primitive()
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
            .into_canonical()
            .unwrap()
            .into_primitive()
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
