// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::BooleanArray;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::VectorExecutor;
use crate::arrow::null_buffer::to_null_buffer;

pub(super) fn to_arrow_bool(
    array: ArrayRef,
    session: &VortexSession,
) -> VortexResult<ArrowArrayRef> {
    let bool_vector = array
        .execute_vector(session)?
        .into_bool_opt()
        .ok_or_else(|| vortex_err!("Failed to convert array to Bool vector"))?;

    let (bits, validity) = bool_vector.into_parts();
    Ok(Arc::new(BooleanArray::new(
        bits.into(),
        to_null_buffer(validity),
    )))
}
