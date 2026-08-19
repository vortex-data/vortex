// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar_fn::fns::mask::MaskReduce;
use vortex_error::VortexResult;

use crate::dense_union::DenseUnion;
use crate::dense_union::DenseUnionArrayExt;
use crate::dense_union::DenseUnionArraySlotsExt;

impl MaskReduce for DenseUnion {
    fn mask(array: ArrayView<'_, Self>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        DenseUnion::try_new(
            array.type_ids().clone().mask(mask.clone())?,
            array.offsets().clone(),
            array.variants().clone(),
            array.iter_children().cloned(),
        )
        .map(|array| Some(array.into_array()))
    }
}
