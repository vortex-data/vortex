// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use crate::Precision;
use crate::arrays::ListViewArray;
use crate::arrays::ListViewVTable;
use crate::dtype::DType;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::stats::StatsSetRef;
use crate::vtable::BaseArrayVTable;

impl BaseArrayVTable<ListViewVTable> for ListViewVTable {
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

    fn array_hash<H: std::hash::Hasher>(
        array: &ListViewArray,
        state: &mut H,
        precision: Precision,
    ) {
        array.dtype.hash(state);
        array.elements().array_hash(state, precision);
        array.offsets().array_hash(state, precision);
        array.sizes().array_hash(state, precision);
        array.validity.array_hash(state, precision);
    }

    fn array_eq(array: &ListViewArray, other: &ListViewArray, precision: Precision) -> bool {
        array.dtype == other.dtype
            && array.elements().array_eq(other.elements(), precision)
            && array.offsets().array_eq(other.offsets(), precision)
            && array.sizes().array_eq(other.sizes(), precision)
            && array.validity.array_eq(&other.validity, precision)
    }
}
