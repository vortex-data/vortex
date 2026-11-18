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
use crate::arrays::dict::TakeReduce;
use crate::arrays::extension::ExtensionArrayExt;

impl TakeReduce for Extension {
    fn take(array: ArrayView<'_, Extension>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let taken_storage = array.storage_array().take(indices.clone())?;
        let dtype = array
            .ext_dtype()
            .with_nullability(taken_storage.dtype().nullability());

        // SAFETY: taking from array preserves selected storage values or produces nulls. The
        // extension dtype uses the result's nullability.
        let result = unsafe { ExtensionArray::new_unchecked(dtype, taken_storage) };
        Ok(Some(result.into_array()))
    }
}

impl TakeExecute for Extension {
    fn take(
        array: ArrayView<'_, Extension>,
        indices: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        <Self as TakeReduce>::take(array, indices)
    }
}
