// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Binary constant-and-column operand dispatch.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::Execution;
use super::Operand;

/// Dispatch a binary strict geometry kernel over constants and columns.
///
/// A null constant or an empty combined validity mask short-circuits to an all-null constant
/// output. Otherwise, `kernel` receives both operand shapes and the mask of rows where both are
/// valid. Two columns are always paired by row index. The kernel remains responsible for physical
/// input interpretation and Vortex output construction.
pub(crate) fn dispatch_binary<K>(
    left: &ArrayRef,
    right: &ArrayRef,
    output_dtype: DType,
    kernel: K,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    K: FnOnce(Execution<2>, &mut ExecutionCtx) -> VortexResult<ArrayRef>,
{
    let len = left.len();
    for operand in [left, right] {
        if operand
            .as_opt::<Constant>()
            .is_some_and(|constant| constant.scalar().is_null())
        {
            return Ok(ConstantArray::new(Scalar::null(output_dtype), len).into_array());
        }
    }

    let (left, right, valid) = match (left.as_opt::<Constant>(), right.as_opt::<Constant>()) {
        (Some(left), Some(right)) => (
            Operand::Constant(left.scalar().clone()),
            Operand::Constant(right.scalar().clone()),
            Mask::new_true(len),
        ),
        (Some(left), None) => (
            Operand::Constant(left.scalar().clone()),
            Operand::Column(right.clone()),
            right.validity()?.execute_mask(len, ctx)?,
        ),
        (None, Some(right)) => (
            Operand::Column(left.clone()),
            Operand::Constant(right.scalar().clone()),
            left.validity()?.execute_mask(len, ctx)?,
        ),
        (None, None) => {
            let left_valid = left.validity()?.execute_mask(len, ctx)?;
            let right_valid = right.validity()?.execute_mask(len, ctx)?;
            (
                Operand::Column(left.clone()),
                Operand::Column(right.clone()),
                &left_valid & &right_valid,
            )
        }
    };

    if len != 0 && valid.all_false() {
        return Ok(ConstantArray::new(Scalar::null(output_dtype), len).into_array());
    }

    kernel(
        Execution {
            operands: [left, right],
            valid,
            len,
            nullability: output_dtype.nullability(),
        },
        ctx,
    )
}
