// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::MaskedArray;
use vortex_array::compute::MaskReduce;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use crate::DateTimePartsArray;
use crate::DateTimePartsVTable;

impl MaskReduce for DateTimePartsVTable {
    fn mask(array: &DateTimePartsArray, validity: &Validity) -> VortexResult<Option<ArrayRef>> {
        let masked_days =
            MaskedArray::try_new(array.days().clone(), validity.clone())?.into_array();
        Ok(Some(
            DateTimePartsArray::try_new(
                array.dtype().as_nullable(),
                masked_days,
                array.seconds().clone(),
                array.subseconds().clone(),
            )?
            .into_array(),
        ))
    }
}
