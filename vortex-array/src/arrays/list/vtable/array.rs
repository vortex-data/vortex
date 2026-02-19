// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use crate::Precision;
use crate::arrays::ListArray;
use crate::arrays::ListVTable;
use crate::dtype::DType;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::stats::StatsSetRef;
use crate::vtable::BaseArrayVTable;

impl BaseArrayVTable<ListVTable> for ListVTable {
    fn len(array: &ListArray) -> usize {
        array.offsets.len().saturating_sub(1)
    }

    fn dtype(array: &ListArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ListArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &ListArray, state: &mut H, precision: Precision) {
        array.dtype.hash(state);
        array.elements.array_hash(state, precision);
        array.offsets.array_hash(state, precision);
        array.validity.array_hash(state, precision);
    }

    fn array_eq(array: &ListArray, other: &ListArray, precision: Precision) -> bool {
        array.dtype == other.dtype
            && array.elements.array_eq(&other.elements, precision)
            && array.offsets.array_eq(&other.offsets, precision)
            && array.validity.array_eq(&other.validity, precision)
    }
}
