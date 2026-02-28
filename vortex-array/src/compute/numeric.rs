// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;
use crate::arrays::ConstantArray;
use crate::arrow::Datum;
use crate::arrow::from_arrow_array_with_len;
use crate::builtins::ArrayBuiltins;
use crate::compute::Options;
use crate::scalar::NumericOperator;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::operators::Operator;

/// Point-wise add two numeric arrays.
///
/// Errs at runtime if the sum would overflow or underflow.
///
/// The result is null at any index that either input is null.
#[deprecated(note = "Use `ArrayBuiltins::binary` instead")]
pub fn add(lhs: &ArrayRef, rhs: &ArrayRef) -> VortexResult<ArrayRef> {
    lhs.to_array().binary(rhs.to_array(), Operator::Add)
}

/// Point-wise add a scalar value to this array on the right-hand-side.
#[deprecated(note = "Use `ArrayBuiltins::binary` instead")]
pub fn add_scalar(lhs: &ArrayRef, rhs: Scalar) -> VortexResult<ArrayRef> {
    lhs.to_array()
        .binary(ConstantArray::new(rhs, lhs.len()).to_array(), Operator::Add)
}

/// Point-wise subtract two numeric arrays.
#[deprecated(note = "Use `ArrayBuiltins::binary` instead")]
pub fn sub(lhs: &ArrayRef, rhs: &ArrayRef) -> VortexResult<ArrayRef> {
    lhs.to_array().binary(rhs.to_array(), Operator::Sub)
}

/// Point-wise subtract a scalar value from this array on the right-hand-side.
#[deprecated(note = "Use `ArrayBuiltins::binary` instead")]
pub fn sub_scalar(lhs: &ArrayRef, rhs: Scalar) -> VortexResult<ArrayRef> {
    lhs.to_array()
        .binary(ConstantArray::new(rhs, lhs.len()).to_array(), Operator::Sub)
}

/// Point-wise multiply two numeric arrays.
#[deprecated(note = "Use `ArrayBuiltins::binary` instead")]
pub fn mul(lhs: &ArrayRef, rhs: &ArrayRef) -> VortexResult<ArrayRef> {
    lhs.to_array().binary(rhs.to_array(), Operator::Mul)
}

/// Point-wise multiply a scalar value into this array on the right-hand-side.
#[deprecated(note = "Use `ArrayBuiltins::binary` instead")]
pub fn mul_scalar(lhs: &ArrayRef, rhs: Scalar) -> VortexResult<ArrayRef> {
    lhs.to_array()
        .binary(ConstantArray::new(rhs, lhs.len()).to_array(), Operator::Mul)
}

/// Point-wise divide two numeric arrays.
#[deprecated(note = "Use `ArrayBuiltins::binary` instead")]
pub fn div(lhs: &ArrayRef, rhs: &ArrayRef) -> VortexResult<ArrayRef> {
    lhs.to_array().binary(rhs.to_array(), Operator::Div)
}

/// Point-wise divide a scalar value into this array on the right-hand-side.
#[deprecated(note = "Use `ArrayBuiltins::binary` instead")]
pub fn div_scalar(lhs: &ArrayRef, rhs: Scalar) -> VortexResult<ArrayRef> {
    lhs.to_array()
        .binary(ConstantArray::new(rhs, lhs.len()).to_array(), Operator::Div)
}

/// Point-wise numeric operation between two arrays of the same type and length.
#[deprecated(note = "Use `ArrayBuiltins::binary` instead")]
pub fn numeric(lhs: &ArrayRef, rhs: &ArrayRef, op: NumericOperator) -> VortexResult<ArrayRef> {
    arrow_numeric(lhs, rhs, op)
}

impl Options for NumericOperator {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Implementation of numeric operations using the Arrow crate.
pub(crate) fn arrow_numeric(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    operator: NumericOperator,
) -> VortexResult<ArrayRef> {
    let nullable = lhs.dtype().is_nullable() || rhs.dtype().is_nullable();
    let len = lhs.len();

    let left = Datum::try_new(lhs)?;
    let right = Datum::try_new_with_target_datatype(rhs, left.data_type())?;

    let array = match operator {
        NumericOperator::Add => arrow_arith::numeric::add(&left, &right)?,
        NumericOperator::Sub => arrow_arith::numeric::sub(&left, &right)?,
        NumericOperator::Mul => arrow_arith::numeric::mul(&left, &right)?,
        NumericOperator::Div => arrow_arith::numeric::div(&left, &right)?,
    };

    from_arrow_array_with_len(array.as_ref(), len, nullable)
}

#[cfg(test)]
#[allow(deprecated)]
mod test {
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::compute::sub_scalar;

    #[test]
    fn test_scalar_subtract_unsigned() {
        let values = buffer![1u16, 2, 3].into_array();
        let result = sub_scalar(&values, 1u16.into()).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([0u16, 1, 2]));
    }

    #[test]
    fn test_scalar_subtract_signed() {
        let values = buffer![1i64, 2, 3].into_array();
        let result = sub_scalar(&values, (-1i64).into()).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([2i64, 3, 4]));
    }

    #[test]
    fn test_scalar_subtract_nullable() {
        let values = PrimitiveArray::from_option_iter([Some(1u16), Some(2), None, Some(3)]);
        let result = sub_scalar(&values.to_array(), Some(1u16).into()).unwrap();
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(0u16), Some(1), None, Some(2)])
        );
    }

    #[test]
    fn test_scalar_subtract_float() {
        let values = buffer![1.0f64, 2.0, 3.0].into_array();
        let to_subtract = -1f64;
        let result = sub_scalar(&values, to_subtract.into()).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([2.0f64, 3.0, 4.0]));
    }

    #[test]
    fn test_scalar_subtract_float_underflow_is_ok() {
        let values = buffer![f32::MIN, 2.0, 3.0].into_array();
        let _results = sub_scalar(&values, 1.0f32.into()).unwrap();
        let _results = sub_scalar(&values, f32::MAX.into()).unwrap();
    }
}
