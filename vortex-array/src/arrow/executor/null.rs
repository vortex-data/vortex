// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::NullArray;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_vector::VectorOps;

use crate::ArrayRef;
use crate::VectorExecutor;

pub(super) fn to_arrow_null(
    array: ArrayRef,
    session: &VortexSession,
) -> VortexResult<ArrowArrayRef> {
    let null_vector = array
        .execute_vector(session)?
        .into_null_opt()
        .ok_or_else(|| vortex_err!("Failed to convert array to Null vector"))?;
    Ok(Arc::new(NullArray::new(null_vector.len())))
}
