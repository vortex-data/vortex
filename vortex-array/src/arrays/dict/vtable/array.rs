// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use super::DictVTable;
use crate::Precision;
use crate::arrays::dict::DictArray;
use crate::dtype::DType;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::stats::StatsSetRef;
use crate::vtable::BaseArrayVTable;

impl BaseArrayVTable<DictVTable> for DictVTable {
    fn len(array: &DictArray) -> usize {
        array.codes.len()
    }

    fn dtype(array: &DictArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &DictArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &DictArray, state: &mut H, precision: Precision) {
        array.dtype.hash(state);
        array.codes.array_hash(state, precision);
        array.values.array_hash(state, precision);
    }

    fn array_eq(array: &DictArray, other: &DictArray, precision: Precision) -> bool {
        array.dtype == other.dtype
            && array.codes.array_eq(&other.codes, precision)
            && array.values.array_eq(&other.values, precision)
    }
}
