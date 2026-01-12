// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::BooleanArray as ArrowBooleanArray;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::VectorExecutor;
use crate::arrays::BoolArray;
use crate::arrow::null_buffer::to_null_buffer;

/// Convert a canonical BoolArray directly to Arrow.
pub fn canonical_bool_to_arrow(array: &BoolArray) -> ArrowArrayRef {
    Arc::new(ArrowBooleanArray::new(
        array.bit_buffer().clone().into(),
        to_null_buffer(array.validity_mask()),
    ))
}

pub(super) fn to_arrow_bool(
    array: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let bool_array = array.execute(ctx)?.into_bool();
    Ok(canonical_bool_to_arrow(&bool_array))
}
