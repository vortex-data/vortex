// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use vortex_dtype::DType;

use crate::arrays::{FixedSizeListArray, FixedSizeListVTable};
use crate::hash::{ArrayEq, ArrayHash};
use crate::stats::StatsSetRef;
use crate::vtable::ArrayVTable;

impl ArrayVTable<FixedSizeListVTable> for FixedSizeListVTable {
    fn len(array: &FixedSizeListArray) -> usize {
        array.len
    }

    fn dtype(array: &FixedSizeListArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &FixedSizeListArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &FixedSizeListArray, state: &mut H) {
        array.dtype.hash(state);
        array.elements().array_hash(state);
        array.list_size().hash(state);
        array.validity.array_hash(state);
        array.len.hash(state);
    }

    fn array_eq(array: &FixedSizeListArray, other: &FixedSizeListArray) -> bool {
        array.dtype == other.dtype
            && array.elements().array_eq(other.elements())
            && array.list_size() == other.list_size()
            && array.validity.array_eq(&other.validity)
            && array.len == other.len
    }
}
