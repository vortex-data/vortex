// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_scalar::Scalar;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::compute::BooleanOperator;
use crate::compute::arrow_boolean;

/// Execute a boolean operation between two arrays.
///
/// This is the entry point for boolean operations from the binary expression.
/// Handles constant-constant directly, otherwise falls back to Arrow.
pub(crate) fn execute_boolean(
    lhs: &dyn Array,
    rhs: &dyn Array,
    op: BooleanOperator,
) -> VortexResult<ArrayRef> {
    if let Some(result) = constant_boolean(lhs, rhs, op)? {
        return Ok(result);
    }
    arrow_boolean(lhs.to_array(), rhs.to_array(), op)
}

fn constant_boolean(
    lhs: &dyn Array,
    rhs: &dyn Array,
    op: BooleanOperator,
) -> VortexResult<Option<ArrayRef>> {
    let (Some(lhs), Some(rhs)) = (
        lhs.as_opt::<ConstantVTable>(),
        rhs.as_opt::<ConstantVTable>(),
    ) else {
        return Ok(None);
    };

    let length = lhs.len();
    let nullable = lhs.dtype().is_nullable() || rhs.dtype().is_nullable();
    let lhs_val = lhs.scalar().as_bool().value();
    let rhs_val = rhs
        .scalar()
        .as_bool_opt()
        .ok_or_else(|| vortex_err!("expected rhs to be boolean"))?
        .value();

    let result = match op {
        BooleanOperator::And => lhs_val.zip(rhs_val).map(|(l, r)| l & r),
        BooleanOperator::AndKleene => match (lhs_val, rhs_val) {
            (Some(false), _) | (_, Some(false)) => Some(false),
            (None, _) | (_, None) => None,
            (Some(l), Some(r)) => Some(l & r),
        },
        BooleanOperator::Or => lhs_val.zip(rhs_val).map(|(l, r)| l | r),
        BooleanOperator::OrKleene => match (lhs_val, rhs_val) {
            (Some(true), _) | (_, Some(true)) => Some(true),
            (None, _) | (_, None) => None,
            (Some(l), Some(r)) => Some(l | r),
        },
    };

    let scalar = result
        .map(|b| Scalar::bool(b, nullable.into()))
        .unwrap_or_else(|| Scalar::null(DType::Bool(nullable.into())));

    Ok(Some(ConstantArray::new(scalar, length).into_array()))
}
