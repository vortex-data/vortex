// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use vortex_dtype::DType;
use vortex_scalar::DecimalType;

use crate::Precision;
use crate::arrays::{DecimalArray, DecimalVTable};
use crate::hash::{ArrayEq, ArrayHash};
use crate::stats::StatsSetRef;
use crate::vtable::ArrayVTable;

impl ArrayVTable<DecimalVTable> for DecimalVTable {
    fn len(array: &DecimalArray) -> usize {
        let divisor = match array.values_type {
            DecimalType::I8 => 1,
            DecimalType::I16 => 2,
            DecimalType::I32 => 4,
            DecimalType::I64 => 8,
            DecimalType::I128 => 16,
            DecimalType::I256 => 32,
        };
        array.values.len() / divisor
    }

    fn dtype(array: &DecimalArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &DecimalArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &DecimalArray, state: &mut H, precision: Precision) {
        array.dtype.hash(state);
        array.values.array_hash(state, precision);
        std::mem::discriminant(&array.values_type).hash(state);
        array.validity.array_hash(state, precision);
    }

    fn array_eq(array: &DecimalArray, other: &DecimalArray, precision: Precision) -> bool {
        array.dtype == other.dtype
            && array.values.array_eq(&other.values, precision)
            && array.values_type == other.values_type
            && array.validity.array_eq(&other.validity, precision)
    }
}
