// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use crate::Precision;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::dtype::DType;
use crate::stats::StatsSetRef;
use crate::vtable::BaseArrayVTable;

impl BaseArrayVTable<ConstantVTable> for ConstantVTable {
    fn len(array: &ConstantArray) -> usize {
        array.len
    }

    fn dtype(array: &ConstantArray) -> &DType {
        array.scalar.dtype()
    }

    fn stats(array: &ConstantArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(
        array: &ConstantArray,
        state: &mut H,
        _precision: Precision,
    ) {
        array.scalar.hash(state);
        array.len.hash(state);
    }

    fn array_eq(array: &ConstantArray, other: &ConstantArray, _precision: Precision) -> bool {
        array.scalar == other.scalar && array.len == other.len
    }
}
