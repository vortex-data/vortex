// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::DictArray;
use crate::arrays::DictVTable;
use crate::arrays::MaskedArray;
use crate::compute::MaskReduce;
use crate::validity::Validity;

impl MaskReduce for DictVTable {
    fn mask(array: &DictArray, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let masked_codes =
            MaskedArray::try_new(array.codes().clone(), Validity::Array(mask.clone()))?
                .into_array();
        // SAFETY: masking codes doesn't change dict invariants
        Ok(Some(unsafe {
            DictArray::new_unchecked(masked_codes, array.values().clone()).into_array()
        }))
    }
}
