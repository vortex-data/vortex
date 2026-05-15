// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Variant;
use crate::arrays::VariantArray;
use crate::arrays::filter::FilterReduce;
use crate::arrays::variant::VariantArrayExt;

impl FilterReduce for Variant {
    fn filter(array: ArrayView<'_, Variant>, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        let core_storage = array.core_storage().filter(mask.clone())?;
        let shredded = array
            .shredded()
            .map(|shredded| shredded.filter(mask.clone()))
            .transpose()?;

        Ok(Some(
            VariantArray::try_new(core_storage, shredded)?.into_array(),
        ))
    }
}
