// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::dtype::Nullability;
use crate::integer;
use crate::integer::IntegerBuffer;
use crate::integer::widened_buffer;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalValue;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::validity::Validity;

enum IntegerOperand {
    Array(IntegerBuffer),
    Constant(DecimalValue, Validity),
}

impl IntegerOperand {
    fn new(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        if let Some(value) = array.as_constant() {
            return Ok(Self::Constant(
                integer::scalar_value(&value)?,
                Validity::from(value.dtype().nullability()),
            ));
        }
        Ok(Self::Array(integer::materialize(array, ctx)?))
    }

    fn validity(&self) -> Validity {
        match self {
            Self::Array(array) => array.validity.clone(),
            Self::Constant(_, validity) => validity.clone(),
        }
    }
}

pub(super) fn compare_integer(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: CompareOperator,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    vortex_ensure!(
        lhs.dtype().eq_ignore_nullability(rhs.dtype()),
        "Cannot compare integer dtypes {} and {}",
        lhs.dtype(),
        rhs.dtype()
    );
    let len = lhs.len();
    let lhs = IntegerOperand::new(lhs, ctx)?;
    let rhs = IntegerOperand::new(rhs, ctx)?;
    let validity = super::compare_validity(lhs.validity(), rhs.validity(), nullability)?;
    let bits = match (lhs, rhs) {
        (IntegerOperand::Array(lhs), IntegerOperand::Array(rhs)) => {
            let width = lhs.values_type.max(rhs.values_type);
            match_each_decimal_value_type!(width, |T| {
                let lhs = widened_buffer::<T>(&lhs);
                let rhs = widened_buffer::<T>(&rhs);
                super::decimal::compare_slices(&lhs, &rhs, op, ctx.allocator())
            })
        }
        (IntegerOperand::Array(array), IntegerOperand::Constant(value, _)) => {
            compare_constant(&array, value, op, ctx)?
        }
        (IntegerOperand::Constant(value, _), IntegerOperand::Array(array)) => {
            compare_constant(&array, value, op.swap(), ctx)?
        }
        (IntegerOperand::Constant(lhs, _), IntegerOperand::Constant(rhs, _)) => {
            BitBuffer::full_in(
                super::ordering_predicate(op)(lhs.as_i256().cmp(&rhs.as_i256())),
                len,
                ctx.allocator().clone(),
            )
        }
    };
    Ok(BoolArray::try_new(bits, validity)?.into_array())
}

fn compare_constant(
    array: &IntegerBuffer,
    value: DecimalValue,
    op: CompareOperator,
    ctx: &ExecutionCtx,
) -> VortexResult<BitBuffer> {
    match_each_decimal_value_type!(array.values_type, |T| {
        let values = Buffer::<T>::from_byte_buffer(array.values.to_host_sync());
        if let Some(value) = value.cast::<T>() {
            return Ok(super::decimal::compare_slice_constant(
                &values,
                value,
                op,
                ctx.allocator(),
            ));
        }
        let negative = value.as_i256() < crate::dtype::i256::ZERO;
        let result = match op {
            CompareOperator::Eq => false,
            CompareOperator::NotEq => true,
            CompareOperator::Lt | CompareOperator::Lte => !negative,
            CompareOperator::Gt | CompareOperator::Gte => negative,
        };
        Ok(BitBuffer::full_in(result, values.len(), ctx.allocator().clone()))
    })
}
