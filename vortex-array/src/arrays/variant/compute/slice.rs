// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Variant;
use crate::arrays::VariantArray;
use crate::arrays::slice::SliceReduce;
use crate::arrays::variant::VariantArrayExt;

impl SliceReduce for Variant {
    fn slice(array: ArrayView<'_, Variant>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let core_storage = array.core_storage().slice(range.clone())?;
        let shredded = array
            .shredded()
            .map(|shredded| shredded.slice(range.clone()))
            .transpose()?;

        Ok(Some(
            VariantArray::try_new(core_storage, shredded)?.into_array(),
        ))
    }
}
