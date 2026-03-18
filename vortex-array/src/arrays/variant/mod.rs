// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod vtable;

pub use self::vtable::Variant;
use crate::ArrayRef;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::stats::ArrayStats;

/// The canonical in-memory representation of variant (semi-structured) data.
///
/// Wraps a single child array that contains the actual variant-encoded data
/// (e.g. a `ParquetVariantArray` or any other variant encoding).
#[derive(Clone, Debug)]
pub struct VariantArray {
    dtype: DType,
    child: ArrayRef,
    pub(super) stats_set: ArrayStats,
}

impl VariantArray {
    /// Creates a new non-nullable VariantArray wrapping the given child.
    pub fn new(child: ArrayRef) -> Self {
        Self::new_nullable(child, Nullability::NonNullable)
    }

    /// Creates a new VariantArray with the given nullability.
    pub fn new_nullable(child: ArrayRef, nullability: Nullability) -> Self {
        Self {
            dtype: DType::Variant(nullability),
            child,
            stats_set: ArrayStats::default(),
        }
    }

    /// Returns a reference to the underlying child array.
    pub fn child(&self) -> &ArrayRef {
        &self.child
    }
}
