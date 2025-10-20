// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use vortex_dtype::DType;

use crate::arrays::extension::{ExtensionArray, ExtensionVTable};
use crate::hash::{ArrayEq, ArrayHash};
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

    fn array_hash<H: std::hash::Hasher>(array: &ExtensionArray, state: &mut H) {
        array.dtype.hash(state);
        array.storage.array_hash(state);
    }

    fn array_eq(array: &ExtensionArray, other: &ExtensionArray) -> bool {
        array.dtype == other.dtype && array.storage.array_eq(&other.storage)
    }
}
