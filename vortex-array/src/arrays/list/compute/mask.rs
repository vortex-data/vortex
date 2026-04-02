// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::List;
use crate::arrays::ListArray;
use crate::scalar_fn::fns::mask::MaskReduce;
use crate::validity::Validity;

impl MaskReduce for List {
    fn mask(array: ArrayView<'_, List>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        ListArray::try_new(
            array.elements().clone(),
            array.offsets().clone(),
            array.validity().and(Validity::Array(mask.clone()))?,
        )
        .map(|a| Some(a.into_array()))
    }
}
