// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Union;
use crate::arrays::UnionArray;
use crate::arrays::slice::SliceReduce;
use crate::arrays::union::UnionArrayExt;

impl SliceReduce for Union {
    fn slice(array: ArrayView<'_, Union>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let type_ids = array.type_ids().slice(range.clone())?;
        let children = array
            .iter_children()
            .map(|child| child.slice(range.clone()))
            .collect::<VortexResult<Vec<ArrayRef>>>()?;

        // SAFETY: Slicing every row-aligned component by the same range preserves all invariants.
        Ok(Some(
            unsafe { UnionArray::new_unchecked(type_ids, array.variants().clone(), children) }
                .into_array(),
        ))
    }
}
