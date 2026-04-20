// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::BooleanArray as ArrowBooleanArray;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::BoolArray;
use crate::arrays::bool::BoolArrayExt;
use crate::arrow::null_buffer::to_null_buffer;

/// Convert a canonical BoolArray directly to Arrow.
pub fn canonical_bool_to_arrow(
    array: &BoolArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    Ok(Arc::new(ArrowBooleanArray::new(
        array.to_bit_buffer().into(),
        to_null_buffer(
            array
                .as_ref()
                .validity()?
                .execute_mask(array.as_ref().len(), ctx)?,
        ),
    )))
}

pub(super) fn to_arrow_bool(
    array: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrowArrayRef> {
    let bool_array = array.execute::<BoolArray>(ctx)?;
    canonical_bool_to_arrow(&bool_array, ctx)
}
