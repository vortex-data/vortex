// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;

use crate::arrays::{PrimitiveArray, PrimitiveVTable};
use crate::stats::StatsSetRef;
use crate::vtable::ArrayVTable;

impl ArrayVTable<PrimitiveVTable> for PrimitiveVTable {
    #[inline]
    fn len(array: &PrimitiveArray) -> usize {
        array.byte_buffer().len() / array.ptype().byte_width()
    }

    #[inline]
    fn dtype(array: &PrimitiveArray) -> &DType {
        &array.dtype
    }

    #[inline]
    fn stats(array: &PrimitiveArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }
}
