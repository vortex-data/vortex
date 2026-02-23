// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::hash::Hasher;

use crate::Precision;
use crate::arrays::PrimitiveArray;
use crate::arrays::PrimitiveVTable;
use crate::dtype::DType;
use crate::hash::ArrayEq;
use crate::hash::ArrayHash;
use crate::stats::StatsSetRef;
use crate::vtable::BaseArrayVTable;

impl BaseArrayVTable<PrimitiveVTable> for PrimitiveVTable {
    fn len(array: &PrimitiveArray) -> usize {
        array.buffer_handle().len() / array.ptype().byte_width()
    }

    fn dtype(array: &PrimitiveArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &PrimitiveArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: Hasher>(array: &PrimitiveArray, state: &mut H, precision: Precision) {
        array.dtype.hash(state);
        array.buffer.array_hash(state, precision);
        array.validity.array_hash(state, precision);
    }

    fn array_eq(array: &PrimitiveArray, other: &PrimitiveArray, precision: Precision) -> bool {
        array.dtype == other.dtype
            && array.buffer.array_eq(&other.buffer, precision)
            && array.validity.array_eq(&other.validity, precision)
    }
}
