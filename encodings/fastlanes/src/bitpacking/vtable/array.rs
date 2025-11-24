// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use vortex_array::stats::StatsSetRef;
use vortex_array::vtable::BaseArrayVTable;
use vortex_array::{ArrayEq, ArrayHash, Precision};
use vortex_dtype::DType;

use crate::{BitPackedArray, BitPackedVTable};

impl BaseArrayVTable<BitPackedVTable> for BitPackedVTable {
    fn len(array: &BitPackedArray) -> usize {
        array.len
    }

    fn dtype(array: &BitPackedArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &BitPackedArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(
        array: &BitPackedArray,
        state: &mut H,
        precision: Precision,
    ) {
        array.offset.hash(state);
        array.len.hash(state);
        array.dtype.hash(state);
        array.bit_width.hash(state);
        array.packed.array_hash(state, precision);
        array.patches.array_hash(state, precision);
        array.validity.array_hash(state, precision);
    }

    fn array_eq(array: &BitPackedArray, other: &BitPackedArray, precision: Precision) -> bool {
        array.offset == other.offset
            && array.len == other.len
            && array.dtype == other.dtype
            && array.bit_width == other.bit_width
            && array.packed.array_eq(&other.packed, precision)
            && array.patches.array_eq(&other.patches, precision)
            && array.validity.array_eq(&other.validity, precision)
    }
}
