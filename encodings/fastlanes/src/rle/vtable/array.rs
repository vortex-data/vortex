// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::Precision;
use vortex_array::dtype::DType;
use vortex_array::stats::StatsSetRef;
use vortex_array::vtable::BaseArrayVTable;

use super::RLEVTable;
use crate::RLEArray;

impl BaseArrayVTable<RLEVTable> for RLEVTable {
    fn len(array: &RLEArray) -> usize {
        array.len()
    }

    fn dtype(array: &RLEArray) -> &DType {
        array.dtype()
    }

    fn stats(array: &RLEArray) -> StatsSetRef<'_> {
        array.stats_set().to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &RLEArray, state: &mut H, precision: Precision) {
        array.dtype().hash(state);
        array.values().array_hash(state, precision);
        array.indices().array_hash(state, precision);
        array.values_idx_offsets().array_hash(state, precision);
        array.offset().hash(state);
        array.len().hash(state);
    }

    fn array_eq(array: &RLEArray, other: &RLEArray, precision: Precision) -> bool {
        array.dtype() == other.dtype()
            && array.values().array_eq(other.values(), precision)
            && array.indices().array_eq(other.indices(), precision)
            && array
                .values_idx_offsets()
                .array_eq(other.values_idx_offsets(), precision)
            && array.offset() == other.offset()
            && array.len() == other.len()
    }
}
