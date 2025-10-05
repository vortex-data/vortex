// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;

use crate::arrays::varbin::{VarBinArray, VarBinVTable};
use crate::stats::StatsSetRef;
use crate::vtable::ArrayVTable;

impl ArrayVTable<VarBinVTable> for VarBinVTable {
    fn len(array: &VarBinArray) -> usize {
        array.offsets().len().saturating_sub(1)
    }

    fn dtype(array: &VarBinArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &VarBinArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }
}
