// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_buffer::NullBuffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ExecutionCtx;
use crate::arrow::null_buffer::to_null_buffer;
use crate::validity::Validity;

pub(super) fn to_arrow_null_buffer(
    validity: &Validity,
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<NullBuffer>> {
    Ok(match validity {
        Validity::NonNullable | Validity::AllValid => None,
        Validity::AllInvalid => Some(NullBuffer::new_null(len)),
        Validity::Array(array) => to_null_buffer(array.clone().execute::<Mask>(ctx)?),
    })
}
