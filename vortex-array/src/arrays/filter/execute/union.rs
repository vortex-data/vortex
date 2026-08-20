// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_mask::Mask;
use vortex_mask::MaskValues;

use crate::ArrayRef;
use crate::arrays::UnionArray;
use crate::arrays::union::UnionArrayExt;
use crate::arrays::union::UnionArraySlotsExt;

pub fn filter_union(array: &UnionArray, mask: &Arc<MaskValues>) -> UnionArray {
    let filter_mask = Mask::Values(Arc::clone(mask));

    let type_ids = array
        .type_ids()
        .filter(filter_mask.clone())
        .vortex_expect("UnionArray type IDs are guaranteed to support filter");

    let children: Vec<ArrayRef> = array
        .iter_children()
        .map(|child| {
            child
                .filter(filter_mask.clone())
                .vortex_expect("UnionArray children are guaranteed to support filter")
        })
        .collect();

    UnionArray::try_new(type_ids, array.variants().clone(), children)
        .vortex_expect("filtered UnionArray children have consistent dtypes and lengths")
}
