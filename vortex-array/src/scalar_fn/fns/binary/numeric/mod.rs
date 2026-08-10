// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native execution of the arithmetic operators (Add/Sub/Mul/Div) of the [`Binary`] scalar
//! function. There is no Arrow fallback.
//!
//! [`Binary`]: super::Binary

mod checked;
mod decimal;
mod primitive;
#[cfg(test)]
mod tests;

use decimal::execute_numeric_decimal;
use primitive::execute_numeric_primitive;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::dtype::DType;
use crate::scalar::NumericOperator;
pub(crate) use crate::scalar::decimal_numeric_result_dtype as numeric_op_result_decimal_dtype;

/// Execute a numeric operation between two arrays.
pub(crate) fn execute_numeric(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: NumericOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    vortex_ensure!(
        lhs.dtype().eq_ignore_nullability(rhs.dtype()),
        "numeric operator requires matching types, got {} and {}",
        lhs.dtype(),
        rhs.dtype()
    );

    let dtype = lhs.dtype();
    vortex_ensure!(
        matches!(dtype, DType::Primitive(..) | DType::Decimal(..)),
        "numeric operator is not supported for dtype {}",
        dtype
    );

    vortex_ensure!(
        lhs.len() == rhs.len(),
        "numeric operator requires equal lengths, got {} and {}",
        lhs.len(),
        rhs.len()
    );

    if lhs.is_empty() {
        return build_empty_result(lhs, rhs, op);
    }

    match dtype {
        DType::Primitive(..) => execute_numeric_primitive(lhs, rhs, op, ctx),
        DType::Decimal(..) => execute_numeric_decimal(lhs, rhs, op, ctx),
        _ => unreachable!("dtype is either Primitive or Decimal"),
    }
}

fn build_empty_result(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: NumericOperator,
) -> VortexResult<ArrayRef> {
    let nullability = lhs.dtype().nullability() | rhs.dtype().nullability();
    let result_dtype = match lhs.dtype() {
        DType::Primitive(..) => lhs.dtype().with_nullability(nullability),
        DType::Decimal(decimal_dtype, _) => DType::Decimal(
            numeric_op_result_decimal_dtype(*decimal_dtype, op)?,
            nullability,
        ),
        _ => unreachable!("dtype is either Primitive or Decimal"),
    };

    Ok(Canonical::empty(&result_dtype).into_array())
}
