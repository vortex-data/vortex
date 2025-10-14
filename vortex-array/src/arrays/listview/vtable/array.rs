// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;

use crate::arrays::{ListViewArray, ListViewVTable};
use crate::stats::StatsSetRef;
use crate::vtable::ArrayVTable;

impl ArrayVTable<ListViewVTable> for ListViewVTable {
    fn len(array: &ListViewArray) -> usize {
        debug_assert_eq!(array.offsets().len(), array.sizes().len());
        array.offsets().len()
    }

    fn dtype(array: &ListViewArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ListViewArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }
}
