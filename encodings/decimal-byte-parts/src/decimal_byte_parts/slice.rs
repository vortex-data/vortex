// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_error::VortexResult;

use super::DecimalBytePartsData;
use crate::DecimalByteParts;

impl SliceReduce for DecimalByteParts {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        // SAFETY: slicing encoded MSP does not change the encoded values
        Ok(Some(unsafe {
            DecimalBytePartsData::new_unchecked(array.msp().slice(range)?, *array.decimal_dtype())
                .into_array()
        }))
    }
}
