// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::SliceReduce;
use vortex_error::VortexResult;

use crate::DecimalBytePartsArray;
use crate::DecimalBytePartsVTable;

impl SliceReduce for DecimalBytePartsVTable {
    fn slice(array: &DecimalBytePartsArray, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        // SAFETY: slicing encoded MSP does not change the encoded values
        Ok(Some(unsafe {
            DecimalBytePartsArray::new_unchecked(array.msp().slice(range)?, *array.decimal_dtype())
                .into_array()
        }))
    }
}
