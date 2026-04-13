// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Struct;
use crate::arrays::StructArray;
use crate::arrays::struct_::StructArrayExt;
use crate::scalar_fn::fns::mask::MaskReduce;
use crate::validity::Validity;

impl MaskReduce for Struct {
    fn mask(array: ArrayView<'_, Struct>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        StructArray::try_new_with_dtype(
            array.unmasked_fields().iter().cloned().collect::<Vec<_>>(),
            array.struct_fields().clone(),
            array.len(),
            array.validity()?.and(Validity::Array(mask.clone()))?,
        )
        .map(|a| Some(a.into_array()))
    }
}
