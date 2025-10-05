// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;

use crate::arrays::varbinview::{VarBinViewArray, VarBinViewVTable};
use crate::stats::StatsSetRef;
use crate::vtable::ArrayVTable;

impl ArrayVTable<VarBinViewVTable> for VarBinViewVTable {
    fn len(array: &VarBinViewArray) -> usize {
        array.views.len()
    }

    fn dtype(array: &VarBinViewArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &VarBinViewArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }
}
