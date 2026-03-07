// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::dtype::DType;
use crate::stats::ArrayStats;

/// Common fields shared by all array types.
///
/// This type will be used during the migration from dynamic array trait to the vtable structs.
/// In the first phase, all arrays will be converted to have a common struct for shared fields.
/// In the second phase, we invert the relationship to have an `Array<V>` where the common fields
/// are hoisted into the generic array struct.
#[derive(Clone, Debug)]
pub struct ArrayCommon {
    len: usize,
    dtype: DType,
    stats: ArrayStats,
}

impl ArrayCommon {
    /// Creates a new `ArrayCommon` with default stats.
    pub fn new(len: usize, dtype: DType) -> Self {
        Self {
            len,
            dtype,
            stats: ArrayStats::default(),
        }
    }

    /// Creates a new `ArrayCommon` with pre-existing stats.
    pub fn new_with_stats(len: usize, dtype: DType, stats: ArrayStats) -> Self {
        Self { len, dtype, stats }
    }

    /// Returns the number of elements in the array.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the array is empty (has zero elements).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the logical [`DType`] of the array.
    #[inline]
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Returns a mutable reference to the [`DType`].
    #[inline]
    pub fn dtype_mut(&mut self) -> &mut DType {
        &mut self.dtype
    }

    /// Sets the number of elements in the array.
    #[inline]
    pub fn set_len(&mut self, len: usize) {
        self.len = len;
    }

    /// Consumes this `ArrayCommon` and returns the owned [`DType`].
    #[inline]
    pub fn into_dtype(self) -> DType {
        self.dtype
    }

    /// Returns the [`ArrayStats`] for this array.
    #[inline]
    pub fn stats(&self) -> &ArrayStats {
        &self.stats
    }
}
