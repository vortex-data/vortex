// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;

use vortex_buffer::Buffer;
use vortex_buffer::lane_ops_indexed::try_map_nullable;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrow::Datum;
use crate::arrow::from_arrow_array_with_len;
use crate::executor::ExecutionCtx;
use crate::match_each_integer_ptype;
use crate::scalar::NumericOperator;
use crate::validity::Validity;

/// Execute a numeric operation between two arrays.
///
/// This is the entry point for numeric operations from the binary expression.
/// Handles constant-constant directly, otherwise falls back to Arrow.
pub(crate) fn execute_numeric(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: NumericOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if let Some(result) = constant_numeric(lhs, rhs, op)? {
        return Ok(result);
    }
    if let Some(result) = primitive_scalar_add(lhs, rhs, op, ctx)? {
        return Ok(result);
    }
    arrow_numeric(lhs, rhs, op, ctx)
}

/// Fast path for `integer_array + integer_scalar` (in either operand order) using the
/// autovectorizing checked-add lane kernel ([`try_map_nullable`]), avoiding the
/// canonicalize-to-Arrow round trip and Arrow's scalar checked-add loop.
///
/// Returns `None` (to defer to [`arrow_numeric`]) whenever the fast path doesn't strictly
/// apply: a non-`Add` op, a shape other than primitive-array × constant-scalar, mismatched
/// or non-integer ptypes, or an overflow at a valid lane. Deferring on overflow keeps the
/// observable behavior identical to the existing Arrow path.
fn primitive_scalar_add(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: NumericOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    if op != NumericOperator::Add {
        return Ok(None);
    }

    // Addition commutes, so accept (array, scalar) in either order.
    let (array, scalar) =
        if let (Some(a), Some(c)) = (lhs.as_opt::<Primitive>(), rhs.as_opt::<Constant>()) {
            (a, c.scalar().clone())
        } else if let (Some(c), Some(a)) = (lhs.as_opt::<Constant>(), rhs.as_opt::<Primitive>()) {
            (a, c.scalar().clone())
        } else {
            return Ok(None);
        };

    let ptype = array.ptype();
    if !ptype.is_int() {
        return Ok(None);
    }
    let Some(scalar) = scalar.as_primitive_opt() else {
        return Ok(None);
    };
    // Only handle the already-matching-type case; Arrow performs any type promotion.
    if scalar.ptype() != ptype {
        return Ok(None);
    }

    let len = array.len();
    let nullable = lhs.dtype().is_nullable() || rhs.dtype().is_nullable();

    match_each_integer_ptype!(ptype, |T| {
        let Some(scalar) = scalar.typed_value::<T>() else {
            // `x + null` is null at every lane.
            return Ok(Some(
                PrimitiveArray::new(array.to_buffer::<T>(), Validity::AllInvalid).into_array(),
            ));
        };

        let validity = array.validity()?.execute_mask(len, ctx)?.to_bit_buffer();
        let mut out: Vec<MaybeUninit<T>> = Vec::with_capacity(len);
        // SAFETY: `try_map_nullable` writes every lane before returning `Ok`.
        unsafe { out.set_len(len) };

        if try_map_nullable(array.as_slice::<T>(), &validity, out.as_mut_slice(), |a| {
            a.checked_add(scalar)
        })
        .is_err()
        {
            // Overflow at a valid lane: defer to Arrow so behavior is unchanged.
            return Ok(None);
        }

        // SAFETY: every lane was initialized since `try_map_nullable` returned `Ok`.
        let values: Vec<T> = unsafe { std::mem::transmute::<Vec<MaybeUninit<T>>, Vec<T>>(out) };
        let validity = if nullable {
            Validity::from(validity)
        } else {
            Validity::NonNullable
        };
        Ok(Some(
            PrimitiveArray::new(Buffer::from(values), validity).into_array(),
        ))
    })
}

/// Implementation of numeric operations using the Arrow crate.
pub(crate) fn arrow_numeric(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    operator: NumericOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let nullable = lhs.dtype().is_nullable() || rhs.dtype().is_nullable();
    let len = lhs.len();

    let left = Datum::try_new(lhs, ctx)?;
    let right = Datum::try_new_with_target_datatype(rhs, left.data_type(), ctx)?;

    let array = match operator {
        NumericOperator::Add => arrow_arith::numeric::add(&left, &right)?,
        NumericOperator::Sub => arrow_arith::numeric::sub(&left, &right)?,
        NumericOperator::Mul => arrow_arith::numeric::mul(&left, &right)?,
        NumericOperator::Div => arrow_arith::numeric::div(&left, &right)?,
    };

    from_arrow_array_with_len(array.as_ref(), len, nullable)
}

fn constant_numeric(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: NumericOperator,
) -> VortexResult<Option<ArrayRef>> {
    let (Some(lhs), Some(rhs)) = (lhs.as_opt::<Constant>(), rhs.as_opt::<Constant>()) else {
        return Ok(None);
    };

    let Some(result) = lhs
        .scalar()
        .as_primitive()
        .checked_binary_numeric(&rhs.scalar().as_primitive(), op)
    else {
        // Overflow detected — fall through to arrow_numeric which uses wrapping arithmetic.
        return Ok(None);
    };

    Ok(Some(ConstantArray::new(result, lhs.len()).into_array()))
}

#[cfg(test)]
mod test {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    use crate::RecursiveCanonical;
    use crate::VortexSessionExecute;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::binary::numeric::ConstantArray;
    use crate::scalar_fn::fns::operators::Operator;

    fn sub_scalar(array: &ArrayRef, scalar: impl Into<Scalar>) -> VortexResult<ArrayRef> {
        array
            .binary(
                ConstantArray::new(scalar, array.len()).into_array(),
                Operator::Sub,
            )
            .and_then(|a| {
                a.execute::<RecursiveCanonical>(&mut LEGACY_SESSION.create_execution_ctx())
            })
            .map(|a| a.0.into_array())
    }

    fn add_scalar(array: &ArrayRef, scalar: impl Into<Scalar>) -> VortexResult<ArrayRef> {
        array
            .binary(
                ConstantArray::new(scalar, array.len()).into_array(),
                Operator::Add,
            )
            .and_then(|a| {
                a.execute::<RecursiveCanonical>(&mut LEGACY_SESSION.create_execution_ctx())
            })
            .map(|a| a.0.into_array())
    }

    #[test]
    fn fast_path_triggers_for_primitive_plus_scalar() {
        use super::primitive_scalar_add;
        use crate::scalar::NumericOperator;

        let array = PrimitiveArray::from_iter([1u32, 2, 3]).into_array();
        let scalar = ConstantArray::new(10u32, 3).into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();

        // Array on the left.
        let out = primitive_scalar_add(&array, &scalar, NumericOperator::Add, &mut ctx)
            .unwrap()
            .expect("fast path should trigger for primitive + scalar");
        assert_arrays_eq!(out, PrimitiveArray::from_iter([11u32, 12, 13]));

        // Scalar on the left (addition commutes).
        let out = primitive_scalar_add(&scalar, &array, NumericOperator::Add, &mut ctx)
            .unwrap()
            .expect("fast path should trigger for scalar + primitive");
        assert_arrays_eq!(out, PrimitiveArray::from_iter([11u32, 12, 13]));

        // Sub does not take the fast path.
        assert!(
            primitive_scalar_add(&array, &scalar, NumericOperator::Sub, &mut ctx)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn add_scalar_matches_expected() {
        let values = buffer![1u32, 2, 3].into_array();
        let result = add_scalar(&values, 10u32).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([11u32, 12, 13]));
    }

    #[test]
    fn add_scalar_preserves_nulls() {
        let array = PrimitiveArray::from_option_iter([Some(1u32), None, Some(3)]).into_array();
        let result = add_scalar(&array, 10u32).unwrap();
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(11u32), None, Some(13)])
        );
    }

    #[test]
    fn add_scalar_overflow_defers_to_arrow() {
        // Overflow at a valid lane: the fast path declines, Arrow's checked add then errors
        // (identical to the pre-existing behavior).
        let values = buffer![1u32, u32::MAX, 3].into_array();
        assert!(add_scalar(&values, 1u32).is_err());
    }

    #[test]
    fn test_scalar_subtract_unsigned() {
        let values = buffer![1u16, 2, 3].into_array();
        let result = sub_scalar(&values, 1u16).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([0u16, 1, 2]));
    }

    #[test]
    fn test_scalar_subtract_signed() {
        let values = buffer![1i64, 2, 3].into_array();
        let result = sub_scalar(&values, -1i64).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([2i64, 3, 4]));
    }

    #[test]
    fn test_scalar_subtract_nullable() {
        let values = PrimitiveArray::from_option_iter([Some(1u16), Some(2), None, Some(3)]);
        let result = sub_scalar(&values.into_array(), Some(1u16)).unwrap();
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(0u16), Some(1), None, Some(2)])
        );
    }

    #[test]
    fn test_scalar_subtract_float() {
        let values = buffer![1.0f64, 2.0, 3.0].into_array();
        let result = sub_scalar(&values, -1f64).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([2.0f64, 3.0, 4.0]));
    }

    #[test]
    fn test_scalar_subtract_float_underflow_is_ok() {
        let values = buffer![f32::MIN, 2.0, 3.0].into_array();
        let _results = sub_scalar(&values, 1.0f32).unwrap();
        let _results = sub_scalar(&values, f32::MAX).unwrap();
    }
}
