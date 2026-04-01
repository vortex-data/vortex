// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::Struct;
use crate::arrays::StructArray;
use crate::scalar_fn::fns::mask::MaskReduce;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

impl MaskReduce for Struct {
    fn mask(array: &StructArray, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        StructArray::try_new_with_dtype(
            array.unmasked_fields().iter().cloned().collect::<Vec<_>>(),
            array.struct_fields().clone(),
            array.len(),
            array
                .validity()
                .clone()
                .and(Validity::Array(mask.clone()))?,
        )
        .map(|a| Some(a.into_array()))
    }
}
