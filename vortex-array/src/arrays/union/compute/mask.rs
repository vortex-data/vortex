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
        let type_ids = array.type_ids().clone().mask(mask.clone())?;
        let children = array.children();

        // SAFETY: Masking type IDs changes only outer validity. Values at newly-null positions are
        // no longer selected, and every row-aligned component retains its length and dtype.
        Ok(Some(
            unsafe { UnionArray::new_unchecked(type_ids, array.variants().clone(), children) }
                .into_array(),
        ))
    }
}
