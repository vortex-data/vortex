// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ListArray;
use crate::arrays::ListVTable;
use crate::compute::MaskReduce;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

impl MaskReduce for ListVTable {
    fn mask(array: &ListArray, validity: &Validity) -> VortexResult<Option<ArrayRef>> {
        ListArray::try_new(
            array.elements().clone(),
            array.offsets().clone(),
            array.validity().clone().and(validity.clone()),
        )
        .map(|a| Some(a.into_array()))
    }
}
