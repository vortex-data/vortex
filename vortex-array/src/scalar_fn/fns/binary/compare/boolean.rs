// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native comparison of boolean arrays using word-wise bit operations.

use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::Constant;
use crate::dtype::Nullability;
use crate::scalar_fn::fns::binary::compare::compare_validity;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::validity::Validity;

enum BoolOperand {
    Array { bits: BitBuffer, validity: Validity },
    Constant { value: bool, validity: Validity },
}

impl BoolOperand {
    fn try_new(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        if let Some(constant) = array.as_opt::<Constant>() {
            let value = constant
                .scalar()
                .as_bool_opt()
                .ok_or_else(|| vortex_err!("expected boolean scalar"))?
                .value()
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

        let array = array.clone().execute::<BoolArray>(ctx)?;
        let validity = array.validity()?;
        Ok(Self::Array {
            bits: array.into_bit_buffer(),
            validity,
        })
    }

    fn validity(&self) -> Validity {
        match self {
            Self::Array { validity, .. } | Self::Constant { validity, .. } => validity.clone(),
        }
    }
}

/// Compare two boolean arrays.
///
/// Values compare as `false < true`; every operator reduces to at most two word-wise passes over
/// the value bit buffers.
pub(super) fn compare_bool(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: CompareOperator,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let len = lhs.len();
    let lhs = BoolOperand::try_new(lhs, ctx)?;
    let rhs = BoolOperand::try_new(rhs, ctx)?;
    let validity = compare_validity(lhs.validity(), rhs.validity(), nullability)?;

    let bits = match (lhs, rhs) {
        (BoolOperand::Array { bits: l, .. }, BoolOperand::Array { bits: r, .. }) => {
            compare_bits(l, r, op)
        }
        (BoolOperand::Array { bits, .. }, BoolOperand::Constant { value, .. }) => {
            compare_bits_constant(bits, value, op)
        }
        (BoolOperand::Constant { value, .. }, BoolOperand::Array { bits, .. }) => {
            compare_bits_constant(bits, value, op.swap())
        }
        (BoolOperand::Constant { value: l, .. }, BoolOperand::Constant { value: r, .. }) => {
            // Unreachable through `execute_compare` (constant-constant is folded there), but
            // cheap to answer anyway.
            let result = super::ordering_predicate(op)(l.cmp(&r));
            BitBuffer::full(result, len)
        }
    };

    Ok(BoolArray::try_new(bits, validity)?.into_array())
}

fn compare_bits(lhs: BitBuffer, rhs: BitBuffer, op: CompareOperator) -> BitBuffer {
    match op {
        CompareOperator::Eq => !(lhs ^ &rhs),
        CompareOperator::NotEq => lhs ^ &rhs,
        // a < b  ⟺  !a & b
        CompareOperator::Lt => rhs.into_bitand_not(&lhs),
        // a <= b  ⟺  !(a & !b)
        CompareOperator::Lte => !lhs.into_bitand_not(&rhs),
        // a > b  ⟺  a & !b
        CompareOperator::Gt => lhs.into_bitand_not(&rhs),
        // a >= b  ⟺  !(!a & b)
        CompareOperator::Gte => !rhs.into_bitand_not(&lhs),
    }
}

/// Compare array bits against a non-null constant: `bits <op> value`.
fn compare_bits_constant(bits: BitBuffer, value: bool, op: CompareOperator) -> BitBuffer {
    let len = bits.len();
    match (op, value) {
        (CompareOperator::Eq, true)
        | (CompareOperator::NotEq, false)
        | (CompareOperator::Gt, false)
        | (CompareOperator::Gte, true) => bits,
        (CompareOperator::Eq, false)
        | (CompareOperator::NotEq, true)
        | (CompareOperator::Lt, true)
        | (CompareOperator::Lte, false) => !bits,
        (CompareOperator::Lt, false) | (CompareOperator::Gt, true) => BitBuffer::new_unset(len),
        (CompareOperator::Lte, true) | (CompareOperator::Gte, false) => BitBuffer::new_set(len),
    }
}
