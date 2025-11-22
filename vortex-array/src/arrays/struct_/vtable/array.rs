// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use vortex_dtype::DType;

use crate::Precision;
use crate::arrays::struct_::{StructArray, StructVTable};
use crate::hash::{ArrayEq, ArrayHash};
use crate::stats::StatsSetRef;
use crate::vtable::BaseArrayVTable;

impl BaseArrayVTable<StructVTable> for StructVTable {
    fn len(array: &StructArray) -> usize {
        array.len
    }

    fn dtype(array: &StructArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &StructArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &StructArray, state: &mut H, precision: Precision) {
        array.len.hash(state);
        array.dtype.hash(state);
        for field in array.fields.iter() {
            field.array_hash(state, precision);
        }
        array.validity.array_hash(state, precision);
    }

    fn array_eq(array: &StructArray, other: &StructArray, precision: Precision) -> bool {
        array.len == other.len
            && array.dtype == other.dtype
            && array.fields.len() == other.fields.len()
            && array
                .fields
                .iter()
                .zip(other.fields.iter())
                .all(|(a, b)| a.array_eq(b, precision))
            && array.validity.array_eq(&other.validity, precision)
    }
}
