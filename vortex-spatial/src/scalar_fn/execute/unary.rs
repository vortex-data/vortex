// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Unary constant/column dispatch for native columnar geometry kernels.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use super::Execution;
use super::Operand;

/// Dispatch a unary strict geometry kernel over a constant or column.
///
/// A null constant or definitively all-null column short-circuits to an all-null constant output.
/// Otherwise, `kernel` receives the operand shape and lazy validity. Kernels that need row-wise
/// validity can materialize a [`vortex_mask::Mask`]; columnar kernels can forward it directly.
pub(crate) fn dispatch_unary<K>(
    array: &ArrayRef,
    output_dtype: DType,
    kernel: K,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    K: FnOnce(Execution<1, Validity>, &mut ExecutionCtx) -> VortexResult<ArrayRef>,
{
    let len = array.len();
    if let Some(constant) = array.as_opt::<Constant>() {
        if constant.scalar().is_null() {
            return Ok(ConstantArray::new(Scalar::null(output_dtype), len).into_array());
        }
        return kernel(
            Execution {
                operands: [Operand::Constant(constant.scalar().clone())],
                valid: Validity::AllValid,
                len,
                nullability: output_dtype.nullability(),
            },
            ctx,
        );
    }

    let valid = array.validity()?;
    if len != 0 && valid.definitely_all_null() {
        return Ok(ConstantArray::new(Scalar::null(output_dtype), len).into_array());
    }
    kernel(
        Execution {
            operands: [Operand::Column(array.clone())],
            valid,
            len,
            nullability: output_dtype.nullability(),
        },
        ctx,
    )
}
