// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::DecimalByteParts;
use crate::decimal_byte_parts::DecimalBytePartsArrayExt;

impl TakeExecute for DecimalByteParts {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let decimal_dtype = *array
            .dtype()
            .as_decimal_opt()
            .vortex_expect("must be a decimal dtype");
        let msp = array.msp().take(indices.clone())?;
        let lower_parts = array
            .lower_parts()
            .into_iter()
            .map(|part| part.take(indices.clone()))
            .collect::<VortexResult<Vec<_>>>()?;
        DecimalByteParts::try_new_parts(msp, lower_parts, decimal_dtype)
            .map(|a| Some(a.into_array()))
    }
}
