// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native comparison of decimal arrays.
//!
//! Both operands share a logical [`DecimalDType`] (equal precision and scale), so comparing the
//! unscaled integer values is sufficient. The physical storage width may differ per operand; when
//! it does, both sides are widened once to the larger storage type before the lane loop.
//!
//! [`DecimalDType`]: crate::dtype::DecimalDType

use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::Constant;
use crate::arrays::DecimalArray;
use crate::arrays::decimal::widened_buffer;
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
    let common = lhs.values_type().max(rhs.values_type());
    match_each_decimal_value_type!(common, |W| {
        let lhs = widened_buffer::<W>(lhs);
        let rhs = widened_buffer::<W>(rhs);
        compare_slices::<W>(&lhs, &rhs, op)
    })
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

fn compare_slices<T: NativeDecimalType>(lhs: &[T], rhs: &[T], op: CompareOperator) -> BitBuffer {
    match op {
        CompareOperator::Eq => collect_zip_bits(lhs, rhs, |a: T, b: T| a == b),
        CompareOperator::NotEq => collect_zip_bits(lhs, rhs, |a: T, b: T| a != b),
        CompareOperator::Gt => collect_zip_bits(lhs, rhs, |a: T, b: T| a > b),
        CompareOperator::Gte => collect_zip_bits(lhs, rhs, |a: T, b: T| a >= b),
        CompareOperator::Lt => collect_zip_bits(lhs, rhs, |a: T, b: T| a < b),
        CompareOperator::Lte => collect_zip_bits(lhs, rhs, |a: T, b: T| a <= b),
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
