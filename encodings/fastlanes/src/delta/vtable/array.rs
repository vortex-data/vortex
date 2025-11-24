// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use vortex_array::stats::StatsSetRef;
use vortex_array::vtable::BaseArrayVTable;
use vortex_array::{ArrayEq, ArrayHash, Precision};
use vortex_dtype::DType;

use super::DeltaVTable;
use crate::DeltaArray;

impl BaseArrayVTable<DeltaVTable> for DeltaVTable {
    fn len(array: &DeltaArray) -> usize {
        array.len()
    }

    fn dtype(array: &DeltaArray) -> &DType {
        array.dtype()
    }

    fn stats(array: &DeltaArray) -> StatsSetRef<'_> {
        array.stats_set().to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &DeltaArray, state: &mut H, precision: Precision) {
        array.offset().hash(state);
        array.len().hash(state);
        array.dtype().hash(state);
        array.bases().array_hash(state, precision);
        array.deltas().array_hash(state, precision);
    }

    fn array_eq(array: &DeltaArray, other: &DeltaArray, precision: Precision) -> bool {
        array.offset() == other.offset()
            && array.len() == other.len()
            && array.dtype() == other.dtype()
            && array.bases().array_eq(other.bases(), precision)
            && array.deltas().array_eq(other.deltas(), precision)
    }
}
