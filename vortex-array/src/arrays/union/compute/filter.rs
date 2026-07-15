// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Union;
use crate::arrays::UnionArray;
use crate::arrays::filter::FilterReduce;
use crate::arrays::union::UnionArrayExt;

impl FilterReduce for Union {
    fn filter(array: ArrayView<'_, Union>, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        let type_ids = array.type_ids().filter(mask.clone())?;
        let children = array
            .iter_children()
            .map(|child| child.filter(mask.clone()))
            .collect::<VortexResult<Vec<ArrayRef>>>()?;

        // SAFETY: Filtering every row-aligned component by the same mask preserves all invariants.
        Ok(Some(
            unsafe { UnionArray::new_unchecked(type_ids, array.variants().clone(), children) }
                .into_array(),
        ))
    }
}
