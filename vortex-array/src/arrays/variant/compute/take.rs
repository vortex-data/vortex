// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Variant;
use crate::arrays::VariantArray;
use crate::arrays::dict::TakeReduce;
use crate::arrays::variant::VariantArrayExt;

impl TakeReduce for Variant {
    fn take(array: ArrayView<'_, Variant>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let core_storage = array.core_storage().take(indices.clone())?;
        let shredded = array
            .shredded()
            .map(|shredded| shredded.take(indices.clone()))
            .transpose()?;

        Ok(Some(
            VariantArray::try_new(core_storage, shredded)?.into_array(),
        ))
    }
}
