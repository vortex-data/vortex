// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Union;
use crate::arrays::UnionArray;
use crate::arrays::union::UnionArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::scalar_fn::fns::mask::MaskReduce;

impl MaskReduce for Union {
    fn mask(array: ArrayView<'_, Union>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        UnionArray::try_new(
            array.type_ids().clone().mask(mask.clone())?,
            array.variants().clone(),
            array.children(),
        )
        .map(|a| Some(a.into_array()))
    }
}
