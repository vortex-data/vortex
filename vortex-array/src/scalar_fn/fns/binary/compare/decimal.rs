// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native comparison of decimal arrays.
//!
//! Both operands share a logical [`DecimalDType`] (equal precision and scale), so comparing the
//! unscaled integer values is sufficient. The physical storage width may differ per operand; when
//! it does, the narrower side is widened lane by lane inside the comparison loop rather than
//! materialized into a widened buffer first.
//!
//! [`DecimalDType`]: crate::dtype::DecimalDType

use std::cmp::Ordering;

use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::Constant;
use crate::arrays::DecimalArray;
use crate::dtype::BigCast;
use crate::dtype::NativeDecimalType;
use crate::dtype::Nullability;
use crate::dtype::i256;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalValue;
use crate::scalar_fn::fns::binary::compare::collect_bits;
use crate::scalar_fn::fns::binary::compare::collect_zip_bits;
use crate::scalar_fn::fns::binary::compare::compare_validity;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::validity::Validity;

enum DecimalOperand {
    Array {
        values: DecimalArray,
        validity: Validity,
    },
    Constant {
        value: DecimalValue,
        validity: Validity,
    },
}

impl DecimalOperand {
    fn try_new(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        if let Some(constant) = array.as_opt::<Constant>() {
            let value = constant
                .scalar()
                .as_decimal()
                .decimal_value()
                .ok_or_else(|| vortex_err!("null constant handled by execute_compare"))?;
            return Ok(Self::Constant {
                value,
                validity: if constant.scalar().dtype().is_nullable() {
                    Validity::AllValid
                } else {
                    Validity::NonNullable
                },
            });
        }

        let values = array.clone().execute::<DecimalArray>(ctx)?;
        let validity = values.validity()?;
        Ok(Self::Array { values, validity })
    }

    fn validity(&self) -> Validity {
        match self {
            Self::Array { validity, .. } | Self::Constant { validity, .. } => validity.clone(),
        }
    }
}

/// Compare two decimal arrays with the same logical decimal dtype.
pub(super) fn compare_decimal(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: CompareOperator,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let len = lhs.len();
    let lhs = DecimalOperand::try_new(lhs, ctx)?;
    let rhs = DecimalOperand::try_new(rhs, ctx)?;
    let validity = compare_validity(lhs.validity(), rhs.validity(), nullability)?;

    let bits = match (lhs, rhs) {
        (DecimalOperand::Array { values: l, .. }, DecimalOperand::Array { values: r, .. }) => {
            compare_decimal_values(&l, &r, op)
        }
        (DecimalOperand::Array { values, .. }, DecimalOperand::Constant { value, .. }) => {
            compare_decimal_constant(&values, value, op)
        }
        (DecimalOperand::Constant { value, .. }, DecimalOperand::Array { values, .. }) => {
            compare_decimal_constant(&values, value, op.swap())
        }
        (DecimalOperand::Constant { value: l, .. }, DecimalOperand::Constant { value: r, .. }) => {
            // Unreachable through `execute_compare` (constant-constant is folded there), but
            // cheap to answer anyway.
            let ordering = l.as_i256().cmp(&r.as_i256());
            BitBuffer::full(super::ordering_predicate(op)(ordering), len)
        }
    };

    Ok(BoolArray::try_new(bits, validity)?.into_array())
}

fn compare_decimal_values(
    lhs: &DecimalArray,
    rhs: &DecimalArray,
    op: CompareOperator,
) -> BitBuffer {
    // Compressed chunks are narrowed independently, so mixed storage widths are the common case
    // rather than the exception. Widening the narrow side is a per-lane sign extension, so it is
    // fused into the comparison instead of materializing a whole widened buffer first.
    match lhs.values_type().cmp(&rhs.values_type()) {
        Ordering::Equal => match_each_decimal_value_type!(lhs.values_type(), |T| {
            compare_slices::<T, T, T>(&lhs.buffer::<T>(), &rhs.buffer::<T>(), op)
        }),
        Ordering::Less => match_each_decimal_value_type!(rhs.values_type(), |W| {
            match_each_decimal_value_type!(lhs.values_type(), |L| {
                compare_slices::<L, W, W>(&lhs.buffer::<L>(), &rhs.buffer::<W>(), op)
            })
        }),
        Ordering::Greater => match_each_decimal_value_type!(lhs.values_type(), |W| {
            match_each_decimal_value_type!(rhs.values_type(), |R| {
                compare_slices::<W, R, W>(&lhs.buffer::<W>(), &rhs.buffer::<R>(), op)
            })
        }),
    }
}

fn compare_decimal_constant(
    array: &DecimalArray,
    constant: DecimalValue,
    op: CompareOperator,
) -> BitBuffer {
    match_each_decimal_value_type!(array.values_type(), |T| {
        match constant.cast::<T>() {
            Some(value) => compare_slice_constant::<T>(&array.buffer::<T>(), value, op),
            None => {
                // The constant does not fit the array's storage type, so it is either greater
                // than every possible array value or less than every possible array value; the
                // sign tells us which.
                let constant_greater = constant.as_i256() > i256::ZERO;
                let result = match op {
                    CompareOperator::Eq => false,
                    CompareOperator::NotEq => true,
                    // array <op> constant
                    CompareOperator::Lt | CompareOperator::Lte => constant_greater,
                    CompareOperator::Gt | CompareOperator::Gte => !constant_greater,
                };
                BitBuffer::full(result, array.len())
            }
        }
    })
}

/// Compare two decimal buffers stored at widths `L` and `R` at the common width `W`.
///
/// `W` must be at least as wide as both `L` and `R`, so that both widening casts are lossless.
fn compare_slices<L, R, W>(lhs: &[L], rhs: &[R], op: CompareOperator) -> BitBuffer
where
    L: NativeDecimalType,
    R: NativeDecimalType,
    W: NativeDecimalType,
{
    /// Widen a lane to the comparison width. Constant-folds away whenever `N` is `W`, and for a
    /// genuine widening it is the sign extension alone: the cast is infallible by construction.
    #[inline]
    fn widen<N: NativeDecimalType, W: NativeDecimalType>(value: N) -> W {
        <W as BigCast>::from(value).vortex_expect("decimal compare widens to a common width")
    }

    macro_rules! zip {
        (| $a:ident, $b:ident | $predicate:expr) => {
            collect_zip_bits(lhs, rhs, |a: L, b: R| {
                let ($a, $b) = (widen::<L, W>(a), widen::<R, W>(b));
                $predicate
            })
        };
    }

    match op {
        CompareOperator::Eq => zip!(|a, b| a == b),
        CompareOperator::NotEq => zip!(|a, b| a != b),
        CompareOperator::Gt => zip!(|a, b| a > b),
        CompareOperator::Gte => zip!(|a, b| a >= b),
        CompareOperator::Lt => zip!(|a, b| a < b),
        CompareOperator::Lte => zip!(|a, b| a <= b),
    }
}

fn compare_slice_constant<T: NativeDecimalType>(
    values: &[T],
    constant: T,
    op: CompareOperator,
) -> BitBuffer {
    match op {
        CompareOperator::Eq => collect_bits(values, |a: T| a == constant),
        CompareOperator::NotEq => collect_bits(values, |a: T| a != constant),
        CompareOperator::Gt => collect_bits(values, |a: T| a > constant),
        CompareOperator::Gte => collect_bits(values, |a: T| a >= constant),
        CompareOperator::Lt => collect_bits(values, |a: T| a < constant),
        CompareOperator::Lte => collect_bits(values, |a: T| a <= constant),
    }
}
