// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::NullArray as ArrowNullArray;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::VectorExecutor;
use crate::arrays::NullArray;

/// Convert a canonical NullArray directly to Arrow.
pub fn canonical_null_to_arrow(array: &NullArray) -> ArrowArrayRef {
    Arc::new(ArrowNullArray::new(array.len()))
}

pub(super) fn to_arrow_null(
    array: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let canonical = array.execute(ctx)?.into_null();
    Ok(canonical_null_to_arrow(&canonical))
}
