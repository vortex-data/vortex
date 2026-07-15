// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Union;
use crate::arrays::UnionArray;
use crate::arrays::dict::TakeReduce;
use crate::arrays::union::UnionArrayExt;

impl TakeReduce for Union {
    fn take(array: ArrayView<'_, Union>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        if indices.dtype().is_nullable() {
            vortex_bail!("Taking UnionArray with nullable indices is not supported yet")
        }

        let type_ids = array.type_ids().take(indices.clone())?;
        let children = array
            .iter_children()
            .map(|child| child.take(indices.clone()))
            .collect::<VortexResult<Vec<ArrayRef>>>()?;

        // SAFETY: Taking every row-aligned component with the same non-null indices preserves all
        // invariants and cannot introduce nulls.
        Ok(Some(
            unsafe { UnionArray::new_unchecked(type_ids, array.variants().clone(), children) }
                .into_array(),
        ))
    }
}
