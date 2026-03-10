// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::DynArray;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::arrays::dict::TakeExecute;

impl TakeExecute for Extension {
    fn take(
        array: &ExtensionArray,
        indices: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let taken_storage = array.storage_array().take(indices.to_array())?;

        Ok(Some(
            // SAFETY: The storage array is taken from an already-valid extension array, which
            // preserves the storage dtype and does not change values.
            unsafe {
                ExtensionArray::new_unchecked(
                    array
                        .ext_dtype()
                        .with_nullability(taken_storage.dtype().nullability()),
                    taken_storage,
                )
            }
            .into_array(),
        ))
    }
}
