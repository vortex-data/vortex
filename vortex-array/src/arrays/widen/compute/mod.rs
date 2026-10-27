// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Operations that run on the values of a [`Widen`](crate::arrays::Widen).

mod cast;
mod compare;
mod fill_null;
mod filter;
mod mask;
pub(crate) mod rules;
mod slice;
mod take;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::WidenArray;
use crate::dtype::DType;

/// Returns `values` when it has `dtype`, and a `Widen` of `dtype` over `values` otherwise.
///
/// `dtype` **must** have the nullability of `values`, because the values hold the validity of the
/// `Widen`. A mask, a fill null or a take with nullable indices can change the nullability of the
/// values, so those callers take the nullability of the result.
pub(super) fn wrap_values(values: ArrayRef, dtype: DType) -> VortexResult<ArrayRef> {
    if values.dtype() == &dtype {
        return Ok(values);
    }

    Ok(WidenArray::try_new(values, dtype)?.into_array())
}
