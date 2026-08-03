// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::NullArray as ArrowNullArray;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::NullArray;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

/// Convert a canonical NullArray directly to Arrow.
pub fn canonical_null_to_arrow(array: &NullArray) -> ArrowArrayRef {
    Arc::new(ArrowNullArray::new(array.len()))
}

pub(super) fn to_arrow_null(
    array: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    vortex_ensure!(
        matches!(array.dtype(), DType::Null),
        "Cannot convert Vortex array with dtype {} to an Arrow Null array",
        array.dtype()
    );
    let null_array = array.execute::<NullArray>(ctx)?;
    Ok(canonical_null_to_arrow(&null_array))
}
