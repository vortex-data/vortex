// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use vortex_dtype::DType;

use crate::Precision;
use crate::arrays::masked::{MaskedArray, MaskedVTable};
use crate::hash::{ArrayEq, ArrayHash};
use crate::stats::StatsSetRef;
use crate::vtable::BaseArrayVTable;

impl BaseArrayVTable<MaskedVTable> for MaskedVTable {
    fn len(array: &MaskedArray) -> usize {
        array.child.len()
    }

    fn dtype(array: &MaskedArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &MaskedArray) -> StatsSetRef<'_> {
        array.stats.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &MaskedArray, state: &mut H, precision: Precision) {
        array.child.array_hash(state, precision);
        array.validity.array_hash(state, precision);
        array.dtype.hash(state);
    }

    fn array_eq(array: &MaskedArray, other: &MaskedArray, precision: Precision) -> bool {
        array.child.array_eq(&other.child, precision)
            && array.validity.array_eq(&other.validity, precision)
            && array.dtype == other.dtype
    }
}
