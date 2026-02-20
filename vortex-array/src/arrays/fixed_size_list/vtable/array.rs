// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use crate::Precision;
use crate::arrays::FixedSizeListArray;
use crate::arrays::FixedSizeListVTable;
use crate::dtype::DType;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::stats::StatsSetRef;
use crate::vtable::BaseArrayVTable;

impl BaseArrayVTable<FixedSizeListVTable> for FixedSizeListVTable {
    fn len(array: &FixedSizeListArray) -> usize {
        array.len
    }

    fn dtype(array: &FixedSizeListArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &FixedSizeListArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(
        array: &FixedSizeListArray,
        state: &mut H,
        precision: Precision,
    ) {
        array.dtype.hash(state);
        array.elements().array_hash(state, precision);
        array.list_size().hash(state);
        array.validity.array_hash(state, precision);
        array.len.hash(state);
    }

    fn array_eq(
        array: &FixedSizeListArray,
        other: &FixedSizeListArray,
        precision: Precision,
    ) -> bool {
        array.dtype == other.dtype
            && array.elements().array_eq(other.elements(), precision)
            && array.list_size() == other.list_size()
            && array.validity.array_eq(&other.validity, precision)
            && array.len == other.len
    }
}
