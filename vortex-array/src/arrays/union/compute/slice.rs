// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use itertools::Itertools;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Union;
use crate::arrays::UnionArray;
use crate::arrays::slice::SliceReduce;
use crate::arrays::union::UnionArrayExt;
use crate::arrays::union::UnionArraySlotsExt;

impl SliceReduce for Union {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let type_ids = array.type_ids().slice(range.clone())?;
        let children: Vec<ArrayRef> = array
            .iter_children()
            .map(|child| child.slice(range.clone()))
            .try_collect()?;

        Ok(Some(
            UnionArray::try_new(type_ids, array.variants().clone(), children)?.into_array(),
        ))
    }
}
