// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use vortex_dtype::DType;

use crate::arrays::{ListViewArray, ListViewVTable};
use crate::hash::{ArrayEq, ArrayHash};
use crate::stats::StatsSetRef;
use crate::vtable::ArrayVTable;

impl ArrayVTable<ListViewVTable> for ListViewVTable {
    fn len(array: &ListViewArray) -> usize {
        debug_assert_eq!(array.offsets().len(), array.sizes().len());
        array.offsets().len()
    }

    fn dtype(array: &ListViewArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ListViewArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &ListViewArray, state: &mut H) {
        array.dtype.hash(state);
        array.elements().array_hash(state);
        array.offsets().array_hash(state);
        array.sizes().array_hash(state);
        array.validity.array_hash(state);
    }

    fn array_eq(array: &ListViewArray, other: &ListViewArray) -> bool {
        array.dtype == other.dtype
            && array.elements().array_eq(other.elements())
            && array.offsets().array_eq(other.offsets())
            && array.sizes().array_eq(other.sizes())
            && array.validity.array_eq(&other.validity)
    }
}
