// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hasher;

use crate::Precision;
use crate::dtype::DType;
use crate::stats::StatsSetRef;
use crate::vtable::VTable;

// FIXME(ngates): inline into VTable when possible
pub trait BaseArrayVTable<V: VTable> {
    fn len(array: &V::Array) -> usize;

    fn dtype(array: &V::Array) -> &DType;

    fn stats(array: &V::Array) -> StatsSetRef<'_>;

    fn array_hash<H: Hasher>(array: &V::Array, state: &mut H, precision: Precision);

    fn array_eq(array: &V::Array, other: &V::Array, precision: Precision) -> bool;
}
