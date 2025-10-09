// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;

use crate::arrays::extension::{
    ExtensionArray,
    ExtensionVTable,
};
use crate::stats::StatsSetRef;
use crate::vtable::ArrayVTable;

impl ArrayVTable<ExtensionVTable> for ExtensionVTable {
    fn len(array: &ExtensionArray) -> usize {
        array.storage.len()
    }

    fn dtype(array: &ExtensionArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ExtensionArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }
}
