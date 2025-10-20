// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use vortex_dtype::DType;

use crate::arrays::varbin::{VarBinArray, VarBinVTable};
use crate::hash::{ArrayEq, ArrayHash};
use crate::stats::StatsSetRef;
use crate::vtable::ArrayVTable;

impl ArrayVTable<VarBinVTable> for VarBinVTable {
    fn len(array: &VarBinArray) -> usize {
        array.offsets().len().saturating_sub(1)
    }

    fn dtype(array: &VarBinArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &VarBinArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &VarBinArray, state: &mut H) {
        array.dtype.hash(state);
        array.bytes().array_hash(state);
        array.offsets().array_hash(state);
        array.validity.array_hash(state);
    }

    fn array_eq(array: &VarBinArray, other: &VarBinArray) -> bool {
        array.dtype == other.dtype
            && array.bytes().array_eq(other.bytes())
            && array.offsets().array_eq(other.offsets())
            && array.validity.array_eq(&other.validity)
    }
}
