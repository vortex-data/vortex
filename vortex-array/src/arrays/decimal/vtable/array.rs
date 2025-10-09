// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::vortex_panic;
use vortex_scalar::DecimalValueType;

use crate::arrays::{
    DecimalArray,
    DecimalVTable,
};
use crate::stats::StatsSetRef;
use crate::vtable::ArrayVTable;

impl ArrayVTable<DecimalVTable> for DecimalVTable {
    fn len(array: &DecimalArray) -> usize {
        let divisor = match array.values_type {
            DecimalValueType::I8 => 1,
            DecimalValueType::I16 => 2,
            DecimalValueType::I32 => 4,
            DecimalValueType::I64 => 8,
            DecimalValueType::I128 => 16,
            DecimalValueType::I256 => 32,
            ty => vortex_panic!("unknown decimal value type {:?}", ty),
        };
        array.values.len() / divisor
    }

    fn dtype(array: &DecimalArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &DecimalArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }
}
