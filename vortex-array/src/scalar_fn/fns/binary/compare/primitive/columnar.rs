// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fused comparison and bit-packing for wide x86 lanes.

use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::operand::PrimitiveOperand;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::binary::compare::collect_bits;
use crate::scalar_fn::fns::binary::compare::collect_zip_bits;
use crate::scalar_fn::fns::binary::compare::compare_validity;
use crate::scalar_fn::fns::operators::CompareOperator;

/// Compare primitive operands with one fused comparison and bit-packing loop.
pub(super) fn compare_primitive(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    match PType::try_from(lhs.dtype())? {
        PType::I64 => compare_primitive_typed::<i64>(lhs, rhs, op, ctx),
        PType::U64 => compare_primitive_typed::<u64>(lhs, rhs, op, ctx),
        PType::F64 => compare_primitive_typed::<f64>(lhs, rhs, op, ctx),
        ptype => vortex_bail!("columnar comparison is not selected for {ptype}"),
    }
}

fn compare_primitive_typed<T: NativePType>(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let len = lhs.len();
    let nullability = Nullability::from(lhs.dtype().is_nullable() || rhs.dtype().is_nullable());
    let lhs = PrimitiveOperand::<T>::try_new(lhs, ctx)?;
    let rhs = PrimitiveOperand::<T>::try_new(rhs, ctx)?;
    if lhs.len() != rhs.len() {
        vortex_bail!(
            "compare operator requires equal lengths, got {} and {}",
            lhs.len(),
            rhs.len()
        );
    }

    let validity = compare_validity(lhs.validity(), rhs.validity(), nullability)?;
    let bits = match (&lhs, &rhs) {
        (
            PrimitiveOperand::Array { values: lhs, .. },
            PrimitiveOperand::Array { values: rhs, .. },
        ) => compare_slices(lhs, rhs, op),
        (
            PrimitiveOperand::Array { values: lhs, .. },
            PrimitiveOperand::Constant { value: rhs, .. },
        ) => compare_slice_constant(lhs, *rhs, op),
        (
            PrimitiveOperand::Constant { value: lhs, .. },
            PrimitiveOperand::Array { values: rhs, .. },
        ) => compare_slice_constant(rhs, *lhs, op.swap()),
        (
            PrimitiveOperand::Constant { value: lhs, .. },
            PrimitiveOperand::Constant { value: rhs, .. },
        ) => BitBuffer::full(apply_op(*lhs, *rhs, op), len),
        (PrimitiveOperand::Null(_), _) | (_, PrimitiveOperand::Null(_)) => {
            return Ok(
                ConstantArray::new(Scalar::null(DType::Bool(Nullability::Nullable)), len)
                    .into_array(),
            );
        }
    };

    Ok(BoolArray::try_new(bits, validity)?.into_array())
}

#[inline(always)]
fn apply_op<T: NativePType>(lhs: T, rhs: T, op: CompareOperator) -> bool {
    match op {
        CompareOperator::Eq => lhs.is_eq(rhs),
        CompareOperator::NotEq => !lhs.is_eq(rhs),
        CompareOperator::Gt => lhs.is_gt(rhs),
        CompareOperator::Gte => lhs.is_ge(rhs),
        CompareOperator::Lt => lhs.is_lt(rhs),
        CompareOperator::Lte => lhs.is_le(rhs),
    }
}

fn compare_slices<T: NativePType>(lhs: &[T], rhs: &[T], op: CompareOperator) -> BitBuffer {
    match op {
        CompareOperator::Eq => collect_zip_bits(lhs, rhs, |lhs: T, rhs: T| lhs.is_eq(rhs)),
        CompareOperator::NotEq => collect_zip_bits(lhs, rhs, |lhs: T, rhs: T| !lhs.is_eq(rhs)),
        CompareOperator::Gt => collect_zip_bits(lhs, rhs, T::is_gt),
        CompareOperator::Gte => collect_zip_bits(lhs, rhs, T::is_ge),
        CompareOperator::Lt => collect_zip_bits(lhs, rhs, T::is_lt),
        CompareOperator::Lte => collect_zip_bits(lhs, rhs, T::is_le),
    }
}

fn compare_slice_constant<T: NativePType>(lhs: &[T], rhs: T, op: CompareOperator) -> BitBuffer {
    match op {
        CompareOperator::Eq => collect_bits(lhs, |lhs: T| lhs.is_eq(rhs)),
        CompareOperator::NotEq => collect_bits(lhs, |lhs: T| !lhs.is_eq(rhs)),
        CompareOperator::Gt => collect_bits(lhs, |lhs: T| lhs.is_gt(rhs)),
        CompareOperator::Gte => collect_bits(lhs, |lhs: T| lhs.is_ge(rhs)),
        CompareOperator::Lt => collect_bits(lhs, |lhs: T| lhs.is_lt(rhs)),
        CompareOperator::Lte => collect_bits(lhs, |lhs: T| lhs.is_le(rhs)),
    }
}
