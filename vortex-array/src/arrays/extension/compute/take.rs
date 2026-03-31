// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::arrays::dict::TakeExecute;
use crate::vtable::Array;

impl TakeExecute for Extension {
    fn take(
        array: &Array<Extension>,
        indices: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let taken_storage = array.storage_array().take(indices.to_array())?;
        Ok(Some(
            ExtensionArray::new(
                array
                    .ext_dtype()
                    .with_nullability(taken_storage.dtype().nullability()),
                taken_storage,
            )
            .into_array(),
        ))
    }
}
