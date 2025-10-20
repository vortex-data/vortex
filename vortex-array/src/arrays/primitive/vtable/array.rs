// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use vortex_dtype::DType;

use crate::arrays::{PrimitiveArray, PrimitiveVTable};
use crate::hash::{ArrayEq, ArrayHash};
use crate::stats::StatsSetRef;
use crate::vtable::ArrayVTable;

impl ArrayVTable<PrimitiveVTable> for PrimitiveVTable {
    fn len(array: &PrimitiveArray) -> usize {
        array.byte_buffer().len() / array.ptype().byte_width()
    }

    fn dtype(array: &PrimitiveArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &PrimitiveArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &PrimitiveArray, state: &mut H) {
        array.dtype.hash(state);
        array.buffer.array_hash(state);
        array.validity.array_hash(state);
    }

    fn array_eq(array: &PrimitiveArray, other: &PrimitiveArray) -> bool {
        array.dtype == other.dtype
            && array.buffer.array_eq(&other.buffer)
            && array.validity.array_eq(&other.validity)
    }
}
