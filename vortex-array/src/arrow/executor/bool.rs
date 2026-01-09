// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::BooleanArray as ArrowBooleanArray;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::VectorExecutor;
use crate::VortexSessionExecute;
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
    session: &VortexSession,
) -> VortexResult<ArrowArrayRef> {
    let mut ctx = session.create_execution_ctx();
    let bool_array = array.execute(&mut ctx)?.into_bool();
    Ok(canonical_bool_to_arrow(&bool_array))
}
