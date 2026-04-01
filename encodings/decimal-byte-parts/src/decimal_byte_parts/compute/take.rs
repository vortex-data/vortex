// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_error::VortexResult;

use super::DecimalBytePartsData;
use crate::DecimalByteParts;

impl TakeExecute for DecimalByteParts {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        DecimalBytePartsData::try_new(array.msp().take(indices.clone())?, *array.decimal_dtype())
            .map(|a| Some(a.into_array()))
    }
}
