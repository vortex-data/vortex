// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod vtable;

use vortex_error::VortexExpect;

pub use self::vtable::Variant;
pub use self::vtable::VariantArray;
use crate::ArrayRef;
use crate::dtype::DType;
use crate::stats::ArrayStats;
use crate::vtable::Array;

/// The canonical in-memory representation of variant (semi-structured) data.
///
/// Wraps a single child array that contains the actual variant-encoded data
/// (e.g. a `ParquetVariantArray` or any other variant encoding).
///
/// Nullability is delegated to the child array: `VariantArray`'s dtype is
/// always the child's dtype. The child's validity determines which rows are
/// null.
#[derive(Clone, Debug)]
pub struct VariantData {
    child: ArrayRef,
    pub(crate) stats_set: ArrayStats,
}

impl VariantData {
    /// Creates a new VariantArray. Nullability comes from the child's dtype.
    pub fn new(child: ArrayRef) -> Self {
        let stats_set = child.statistics().to_array_stats();
        Self { child, stats_set }
    }

    /// Returns the length of this array.
    pub fn len(&self) -> usize {
        self.child.len()
    }

    /// Returns the [`DType`] of this array.
    pub fn dtype(&self) -> &DType {
        self.child.dtype()
    }

    /// Returns `true` if this array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a reference to the underlying child array.
    pub fn child(&self) -> &ArrayRef {
        &self.child
    }
}

impl Array<Variant> {
    /// Creates a new `VariantArray`.
    pub fn new(child: ArrayRef) -> Self {
        Array::try_from_data(VariantData::new(child)).vortex_expect("VariantData is always valid")
    }
}
