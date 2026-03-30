// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar_fn::fns::mask::MaskReduce;
use vortex_error::VortexResult;

use crate::DateTimeParts;
use crate::DateTimePartsArray;
use crate::DateTimePartsArrayParts;
use crate::DateTimePartsData;

impl MaskReduce for DateTimeParts {
    fn mask(array: &DateTimePartsArray, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let DateTimePartsArrayParts {
            dtype,
            days,
            seconds,
            subseconds,
        } = array.clone().into_data().into_parts();
        let masked_days = days.mask(mask.clone())?;
        Ok(Some(
            DateTimePartsData::try_new(dtype.as_nullable(), masked_days, seconds, subseconds)?
                .into_array(),
        ))
    }
}
