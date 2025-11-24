// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use vortex_array::stats::StatsSetRef;
use vortex_array::vtable::BaseArrayVTable;
use vortex_array::{ArrayEq, ArrayHash, Precision};
use vortex_dtype::DType;

use super::FoRVTable;
use crate::FoRArray;

impl BaseArrayVTable<FoRVTable> for FoRVTable {
    fn len(array: &FoRArray) -> usize {
        array.encoded().len()
    }

    fn dtype(array: &FoRArray) -> &DType {
        array.reference_scalar().dtype()
    }

    fn stats(array: &FoRArray) -> StatsSetRef<'_> {
        array.stats_set().to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &FoRArray, state: &mut H, precision: Precision) {
        array.encoded().array_hash(state, precision);
        array.reference_scalar().hash(state);
    }

    fn array_eq(array: &FoRArray, other: &FoRArray, precision: Precision) -> bool {
        array.encoded().array_eq(other.encoded(), precision)
            && array.reference_scalar() == other.reference_scalar()
    }
}
