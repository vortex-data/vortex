// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::filter::FilterReduce;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::dense_union::DenseUnion;
use crate::dense_union::DenseUnionArrayExt;
use crate::dense_union::DenseUnionArraySlotsExt;

impl FilterReduce for DenseUnion {
    fn filter(array: ArrayView<'_, Self>, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        DenseUnion::try_new(
            array.type_ids().filter(mask.clone())?,
            array.offsets().filter(mask.clone())?,
            array.variants().clone(),
            array.iter_children().cloned(),
        )
        .map(|array| Some(array.into_array()))
    }
}
