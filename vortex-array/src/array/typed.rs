// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::dtype::DType;
use crate::stats::{ArrayStats, StatsSetRef};
use crate::vtable::VTable;
use crate::ArrayRef;
use std::ops::Deref;

/// The typed representation of a Vortex array, pairing a vtable with its array data and other
/// common fields.
#[derive(Debug, Clone)]
pub struct Array<V: VTable> {
    vtable: V,
    len: usize,
    dtype: DType,
    data: V::Array,
    child_slots: Vec<Option<ArrayRef>>,
    stats: ArrayStats,
}

/// Deref to the inner array data for convenient access to typed accessors.
impl<V: VTable> Deref for Array<V> {
    type Target = V::Array;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<V: VTable> Array<V> {
    /// Returns the length of the array.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the array is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the dtype of the array.
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Returns the stats of the array.
    pub fn stats(&self) -> &ArrayStats {
        &self.stats
    }
}
