// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::arrays::dict::TakeExecute;

impl TakeExecute for Extension {
    fn take(
        array: ArrayView<'_, Extension>,
        indices: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let taken_storage = array.storage_array().take(indices.clone())?;
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
