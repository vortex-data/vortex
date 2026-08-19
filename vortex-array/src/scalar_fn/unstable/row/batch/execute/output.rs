// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

use super::super::Batch;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::scalar_fn::ScalarFnId;

impl Batch {
    pub(super) fn all_null(&self) -> ArrayRef {
        ConstantArray::new(Scalar::null(self.result_dtype.clone()), self.row_count).into_array()
    }

    /// Validate the finished output and apply the row function's logical outer nullability.
    pub(super) fn finalize_output(
        &self,
        values: ArrayRef,
        expected_len: usize,
    ) -> VortexResult<ArrayRef> {
        validate_output(self.id, &self.result_dtype, expected_len, &values)?;

        cast_output_nullability(&self.result_dtype, values)
    }

    /// Validate the output from a row function before batch validity is attached.
    pub(super) fn validate_kernel_output(
        &self,
        values: ArrayRef,
        expected_len: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        finalize_kernel_output(self.id, &self.output_dtype, expected_len, values, ctx)
    }
}

/// Validate the output produced directly by a row function.
///
/// `values` **must** contain `expected_len` rows. Its dtype must match `result_dtype` except for
/// outer nullability, and every produced row **must** be valid. Batch execution owns strict null
/// propagation and attaches input-derived validity only after this boundary.
pub(crate) fn finalize_kernel_output(
    id: ScalarFnId,
    result_dtype: &DType,
    expected_len: usize,
    values: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    validate_output(id, result_dtype, expected_len, &values)?;
    vortex_ensure!(
        values.all_valid(ctx)?,
        "the {id} row kernel must produce only valid rows, got at least one null row",
    );

    cast_output_nullability(result_dtype, values)
}

/// Validate an output's shape and logical dtype without executing an outer-nullability cast.
fn validate_output(
    id: ScalarFnId,
    result_dtype: &DType,
    expected_len: usize,
    values: &ArrayRef,
) -> VortexResult<()> {
    vortex_ensure_eq!(
        values.len(),
        expected_len,
        "the {id} kernel output must contain {expected_len} rows, got {}",
        values.len(),
    );
    let values_with_result_nullability =
        values.dtype().with_nullability(result_dtype.nullability());
    vortex_ensure!(
        values_with_result_nullability == *result_dtype,
        "the {id} output dtype must match {result_dtype} except for outer nullability, got {}",
        values.dtype(),
    );

    Ok(())
}

/// Cast only the outer output nullability after validation accepts every other dtype component.
///
/// This changes no logical dtype component other than outer nullability. An encoding that cannot
/// rewrite its nullability directly can still retain a lazy cast until execution.
fn cast_output_nullability(result_dtype: &DType, values: ArrayRef) -> VortexResult<ArrayRef> {
    if values.dtype() == result_dtype {
        Ok(values)
    } else {
        values.cast(result_dtype.clone())
    }
}
