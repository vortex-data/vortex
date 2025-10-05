// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;

use crate::arrays::masked::{MaskedArray, MaskedVTable};
use crate::stats::StatsSetRef;
use crate::vtable::ArrayVTable;

impl ArrayVTable<MaskedVTable> for MaskedVTable {
    fn len(array: &MaskedArray) -> usize {
        array.child.len()
    }

    fn dtype(array: &MaskedArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &MaskedArray) -> StatsSetRef<'_> {
        array.stats.to_ref(array.as_ref())
    }
}
