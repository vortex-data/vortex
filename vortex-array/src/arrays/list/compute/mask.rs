// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::list::ListArray;
use crate::arrays::list::ListVTable;
use crate::scalar_fn::fns::mask::MaskReduce;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

impl MaskReduce for ListVTable {
    fn mask(array: &ListArray, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        ListArray::try_new(
            array.elements().clone(),
            array.offsets().clone(),
            array
                .validity()
                .clone()
                .and(Validity::Array(mask.clone()))?,
        )
        .map(|a| Some(a.into_array()))
    }
}
