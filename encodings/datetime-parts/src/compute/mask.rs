// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar_fn::fns::mask::MaskReduce;
use vortex_error::VortexResult;

use crate::DateTimeParts;
use crate::DateTimePartsData;

impl MaskReduce for DateTimeParts {
    fn mask(array: ArrayView<'_, Self>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let masked_days = array.days().clone().mask(mask.clone())?;
        Ok(Some(
            DateTimePartsData::try_new(
                array.dtype().as_nullable(),
                masked_days,
                array.seconds().clone(),
                array.subseconds().clone(),
            )?
            .into_array(),
        ))
    }
}
