// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use vortex_dtype::DType;

use crate::Precision;
use crate::arrays::BoolArray;
use crate::arrays::BoolVTable;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::stats::StatsSetRef;
use crate::vtable::BaseArrayVTable;

impl BaseArrayVTable<BoolVTable> for BoolVTable {
    fn len(array: &BoolArray) -> usize {
        array.bits.len()
    }

    fn dtype(array: &BoolArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &BoolArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &BoolArray, state: &mut H, precision: Precision) {
        array.dtype.hash(state);
        array.bit_buffer().array_hash(state, precision);
        array.validity.array_hash(state, precision);
    }

    fn array_eq(array: &BoolArray, other: &BoolArray, precision: Precision) -> bool {
        if array.dtype != other.dtype {
            return false;
        }
        array.bit_buffer().array_eq(other.bit_buffer(), precision)
            && array.validity.array_eq(&other.validity, precision)
    }
}
