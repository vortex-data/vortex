// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Not;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::compute::MaskReduce;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use crate::DateTimePartsArray;
use crate::DateTimePartsArrayParts;
use crate::DateTimePartsVTable;

impl MaskReduce for DateTimePartsVTable {
    fn mask(array: &DateTimePartsArray, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let DateTimePartsArrayParts {
            dtype,
            days,
            seconds,
            subseconds,
        } = array.clone().into_parts();
        let vortex_mask = Validity::Array(mask.clone()).to_mask(days.len()).not();
        let masked_days = days.mask(&vortex_mask)?;
        Ok(Some(
            DateTimePartsArray::try_new(dtype.as_nullable(), masked_days, seconds, subseconds)?
                .into_array(),
        ))
    }
}
