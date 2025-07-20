// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;

use crate::stats::StatsSetRef;
use crate::vtable::VTable;

/// VTable for basic array operations.
pub trait ArrayVTable<V: VTable> {
    /// Returns the length of the array.
    fn len(array: &V::Array) -> usize;

    /// Returns the data type of the array.
    fn dtype(array: &V::Array) -> &DType;

    /// Returns a reference to the array's statistics.
    fn stats(array: &V::Array) -> StatsSetRef<'_>;
}
