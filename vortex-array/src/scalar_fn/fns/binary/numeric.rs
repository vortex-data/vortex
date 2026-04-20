// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::CheckedDiv;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::arrow::Datum;
use crate::arrow::from_arrow_array_with_len;
#[expect(deprecated)]
use crate::canonical::ToCanonical;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::match_each_native_ptype;
use crate::scalar::NumericOperator;
use crate::validity::Validity;

/// Execute a numeric operation between two arrays.
///
/// This is the entry point for numeric operations from the binary expression.
/// Handles constant-constant directly, otherwise falls back to Arrow (or the
/// dedicated SafeDiv kernel).
pub(crate) fn execute_numeric(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: NumericOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if let Some(result) = constant_numeric(lhs, rhs, op)? {
        return Ok(result);
    }
    match op {
        NumericOperator::SafeDiv => arrow_safe_div(lhs, rhs, ctx),
        _ => arrow_numeric(lhs, rhs, op),
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
        NumericOperator::SafeDiv => {
            unreachable!("SafeDiv is dispatched to arrow_safe_div in execute_numeric")
        }
    };

    from_arrow_array_with_len(array.as_ref(), len, nullable)
}

/// Element-wise SafeDiv kernel for primitive numeric arrays.
///
/// Single-pass hand-written kernel: integer `x / 0` yields `0` (not an error), float `x / 0.0`
/// yields `0.0` (not `Inf`/`NaN`). Integer overflow (e.g. `i32::MIN / -1`) still errs, matching
/// [`NumericOperator::Div`]. Nulls propagate from either side.
fn arrow_safe_div(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if !matches!(lhs.dtype(), DType::Primitive(..)) {
        vortex_bail!(
            "safe_div is only implemented for primitive arrays, got {}",
            lhs.dtype()
        );
    }

    #[expect(deprecated)]
    let lhs = lhs.to_primitive();
    #[expect(deprecated)]
    let rhs = rhs.to_primitive();

    // `Validity::and` on two `Validity::Array` variants returns a lazy expression; materialize
    // it here so downstream reads see the combined bitmap rather than the unevaluated AND node.
    let validity = match lhs.validity()?.and(rhs.validity()?)? {
        Validity::Array(v) => Validity::Array(v.execute::<BoolArray>(ctx)?.into_array()),
        other => other,
    };
    let ptype = lhs.ptype();

    match_each_native_ptype!(ptype,
        integral: |T| {
            safe_div_integral::<T>(lhs.as_slice::<T>(), rhs.as_slice::<T>(), validity)
        },
        floating: |T| {
            Ok(safe_div_floating::<T>(lhs.as_slice::<T>(), rhs.as_slice::<T>(), validity))
        }
    )
}

fn safe_div_integral<T: NativePType + CheckedDiv>(
    lhs: &[T],
    rhs: &[T],
    validity: Validity,
) -> VortexResult<ArrayRef> {
    let zero = T::zero();
    let mut buffer = BufferMut::<T>::with_capacity(lhs.len());
    for (l, r) in lhs.iter().zip(rhs) {
        let value = if *r == zero {
            zero
        } else {
            l.checked_div(r)
                .ok_or_else(|| vortex_err!("integer overflow in safe_div"))?
        };
        buffer.push(value);
    }
    Ok(PrimitiveArray::new(buffer.freeze(), validity).into_array())
}

fn safe_div_floating<T: NativePType>(lhs: &[T], rhs: &[T], validity: Validity) -> ArrayRef {
    let zero = T::zero();
    let buffer = BufferMut::<T>::from_trusted_len_iter(
        lhs.iter()
            .zip(rhs)
            .map(|(&l, &r)| if r == zero { zero } else { l / r }),
    )
    .freeze();
    PrimitiveArray::new(buffer, validity).into_array()
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

    fn safe_div(lhs: ArrayRef, rhs: ArrayRef) -> VortexResult<ArrayRef> {
        lhs.binary(rhs, Operator::SafeDiv)
            .and_then(|a| {
                a.execute::<RecursiveCanonical>(&mut LEGACY_SESSION.create_execution_ctx())
            })
            .map(|a| a.0.into_array())
    }

    #[test]
    fn test_safe_div_integer_by_zero_yields_zero() {
        let lhs = buffer![10i32, 20, 30].into_array();
        let rhs = buffer![2i32, 0, 5].into_array();
        let result = safe_div(lhs, rhs).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([5i32, 0, 6]));
    }

    #[test]
    fn test_safe_div_float_by_zero_is_zero_not_inf() {
        let lhs = buffer![1.0f64, 2.0, 3.0].into_array();
        let rhs = buffer![2.0f64, 0.0, 6.0].into_array();
        let result = safe_div(lhs, rhs).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([0.5f64, 0.0, 0.5]));
    }

    #[test]
    fn test_safe_div_nullable_propagation() {
        let lhs =
            PrimitiveArray::from_option_iter([Some(10i32), Some(20), None, Some(40)]).into_array();
        let rhs =
            PrimitiveArray::from_option_iter([Some(2i32), Some(0), Some(0), None]).into_array();
        // Expected: 10/2=5, 20/0=0 (safe), null lhs -> null, null rhs -> null.
        let result = safe_div(lhs, rhs).unwrap();
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_option_iter([Some(5i32), Some(0), None, None])
        );
    }

    #[test]
    fn test_safe_div_constant_zero_rhs() {
        // Constant-constant fast path: checked_binary_numeric in constant_numeric().
        let lhs = buffer![7i32, 8, 9].into_array();
        let rhs = ConstantArray::new(0i32, lhs.len()).into_array();
        let result = safe_div(lhs, rhs).unwrap();
        assert_arrays_eq!(result, PrimitiveArray::from_iter([0i32, 0, 0]));
    }
}
