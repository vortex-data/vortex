// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::DecimalByteParts;
use crate::decimal_byte_parts::DecimalBytePartsArrayExt;

impl SliceReduce for DecimalByteParts {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let decimal_dtype = *array
            .dtype()
            .as_decimal_opt()
            .vortex_expect("must be a decimal dtype");
        let msp = array.msp().slice(range.clone())?;
        let lower_parts = array
            .lower_parts()
            .into_iter()
            .map(|part| part.slice(range.clone()))
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(Some(
            DecimalByteParts::try_new_parts(msp, lower_parts, decimal_dtype)?.into_array(),
        ))
    }
}
