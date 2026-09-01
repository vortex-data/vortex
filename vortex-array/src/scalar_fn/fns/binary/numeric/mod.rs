// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native execution of the arithmetic operators (Add/Sub/Mul/Div) of the [`Binary`] scalar
//! function. There is no Arrow fallback.
//!
//! The primitive widths are computed by a
//! [`RowFn`](crate::scalar_fn::unstable::row::RowFn), which owns null handling, constants, and
//! validity for them; see [`row`]. Decimal keeps its own columnar implementation in [`decimal`].
//!
//! [`Binary`]: super::Binary

mod checked;
mod decimal;
mod primitive;
mod row;

use decimal::execute_numeric_decimal;
use row::execute_numeric_primitive;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::PType;
use crate::scalar::NumericOperator;
use crate::scalar::decimal_multiply_result_dtype;
pub(crate) use crate::scalar::decimal_numeric_result_dtype as numeric_op_result_decimal_dtype;

pub(super) fn numeric_return_dtype(
    lhs: &DType,
    rhs: &DType,
    op: NumericOperator,
) -> VortexResult<DType> {
    let nullability = lhs.nullability() | rhs.nullability();

    if lhs.is_primitive() && lhs.eq_ignore_nullability(rhs) {
        return Ok(lhs.with_nullability(nullability));
    }

    if op == NumericOperator::Mul
        && (lhs.is_decimal() || rhs.is_decimal())
        && let (Some(lhs), Some(rhs)) = (
            decimal_multiply_operand_dtype(lhs),
            decimal_multiply_operand_dtype(rhs),
        )
    {
        return Ok(DType::Decimal(
            decimal_multiply_result_dtype(lhs, rhs)?,
            nullability,
        ));
    }

    if let (DType::Decimal(lhs_decimal, _), DType::Decimal(rhs_decimal, _)) = (lhs, rhs)
        && lhs_decimal == rhs_decimal
    {
        return Ok(DType::Decimal(
            numeric_op_result_decimal_dtype(*lhs_decimal, op)?,
            nullability,
        ));
    }

    vortex_bail!(
        "incompatible types for arithmetic operation: {} {}",
        lhs,
        rhs
    )
}

pub(super) fn decimal_multiply_operand_dtype(dtype: &DType) -> Option<DecimalDType> {
    match dtype {
        DType::Decimal(dtype, _) => Some(*dtype),
        DType::Primitive(ptype, _) if ptype.is_signed_int() => {
            let precision = match ptype {
                PType::I8 => 3,
                PType::I16 => 5,
                PType::I32 => 10,
                PType::I64 => 19,
                _ => unreachable!("ptype is a signed integer"),
            };
            Some(DecimalDType::new(precision, 0))
        }
        _ => None,
    }
}

/// Execute a numeric operation between two arrays.
pub(crate) fn execute_numeric(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: NumericOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    vortex_ensure!(
        lhs.len() == rhs.len(),
        "numeric operator requires equal lengths, got {} and {}",
        lhs.len(),
        rhs.len()
    );

    let result_dtype = numeric_return_dtype(lhs.dtype(), rhs.dtype(), op)?;

    if lhs.is_empty() {
        return Ok(Canonical::empty(&result_dtype).into_array());
    }

    match result_dtype {
        DType::Primitive(..) => execute_numeric_primitive(lhs, rhs, op, ctx),
        DType::Decimal(decimal_dtype, _) => {
            execute_numeric_decimal(lhs, rhs, op, decimal_dtype, ctx)
        }
        _ => unreachable!("numeric result is either Primitive or Decimal"),
    }
}

#[cfg(test)]
mod tests;
