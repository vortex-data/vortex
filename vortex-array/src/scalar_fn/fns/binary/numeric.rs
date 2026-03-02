// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::arrow::Datum;
use crate::arrow::from_arrow_array_with_len;
use crate::scalar::NumericOperator;

/// Execute a numeric operation between two arrays.
///
/// This is the entry point for numeric operations from the binary expression.
/// Handles constant-constant directly, otherwise falls back to Arrow.
pub(crate) fn execute_numeric(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: NumericOperator,
) -> VortexResult<ArrayRef> {
    if let Some(result) = constant_numeric(lhs, rhs, op)? {
        return Ok(result);
    }
    arrow_numeric(lhs, rhs, op)
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

fn constant_numeric(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: NumericOperator,
) -> VortexResult<Option<ArrayRef>> {
    let (Some(lhs), Some(rhs)) = (
        lhs.as_opt::<ConstantVTable>(),
        rhs.as_opt::<ConstantVTable>(),
    ) else {
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
