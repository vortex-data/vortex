// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use crate::Precision;
use crate::arrays::varbin::VarBinArray;
use crate::arrays::varbin::VarBinVTable;
use crate::dtype::DType;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::stats::StatsSetRef;
use crate::vtable::BaseArrayVTable;

impl BaseArrayVTable<VarBinVTable> for VarBinVTable {
    fn len(array: &VarBinArray) -> usize {
        array.offsets().len().saturating_sub(1)
    }

    fn dtype(array: &VarBinArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &VarBinArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &VarBinArray, state: &mut H, precision: Precision) {
        array.dtype.hash(state);
        array.bytes().array_hash(state, precision);
        array.offsets().array_hash(state, precision);
        array.validity.array_hash(state, precision);
    }

    fn array_eq(array: &VarBinArray, other: &VarBinArray, precision: Precision) -> bool {
        array.dtype == other.dtype
            && array.bytes().array_eq(other.bytes(), precision)
            && array.offsets().array_eq(other.offsets(), precision)
            && array.validity.array_eq(&other.validity, precision)
    }
}
