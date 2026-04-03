// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_error::VortexResult;

use crate::encodings::turboquant::TurboQuant;

impl TakeExecute for TurboQuant {
    fn take(
        array: ArrayView<'_, TurboQuant>,
        indices: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // FSL children handle per-row take natively.
        let taken_codes = array.codes().take(indices.clone())?;
        let taken_norms = array.norms().take(indices.clone())?;

        Ok(Some(
            TurboQuant::try_new_array(
                array.dtype().clone(),
                taken_codes,
                taken_norms,
                array.centroids().clone(),
                array.rotation_signs().clone(),
            )?
            .into_array(),
        ))
    }
}
