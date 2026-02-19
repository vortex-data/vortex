// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::compute::MaskReduce;
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
        let masked_days = days.mask(mask.clone())?;
        Ok(Some(
            DateTimePartsArray::try_new(dtype.as_nullable(), masked_days, seconds, subseconds)?
                .into_array(),
        ))
    }
}
