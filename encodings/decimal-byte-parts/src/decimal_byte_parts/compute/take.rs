// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::DynArray;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_error::VortexResult;

use crate::DecimalByteParts;
use crate::DecimalBytePartsArray;

impl TakeExecute for DecimalByteParts {
    fn take(
        array: &DecimalBytePartsArray,
        indices: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        DecimalBytePartsArray::try_new(
            array.msp().take(indices.to_array())?,
            *array.decimal_dtype(),
        )
        .map(|a| Some(a.into_array()))
    }
}
