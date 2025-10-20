// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use vortex_dtype::DType;

use crate::arrays::{ListArray, ListVTable};
use crate::hash::{ArrayEq, ArrayHash};
use crate::stats::StatsSetRef;
use crate::vtable::ArrayVTable;

impl ArrayVTable<ListVTable> for ListVTable {
    fn len(array: &ListArray) -> usize {
        array.offsets.len().saturating_sub(1)
    }

    fn dtype(array: &ListArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ListArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &ListArray, state: &mut H) {
        array.dtype.hash(state);
        array.elements.array_hash(state);
        array.offsets.array_hash(state);
        array.validity.array_hash(state);
    }

    fn array_eq(array: &ListArray, other: &ListArray) -> bool {
        array.dtype == other.dtype
            && array.elements.array_eq(&other.elements)
            && array.offsets.array_eq(&other.offsets)
            && array.validity.array_eq(&other.validity)
    }
}
