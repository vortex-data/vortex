// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::compute::arrow_numeric;
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
