// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod vtable;

pub use self::vtable::Variant;
use crate::ArrayRef;

/// The canonical in-memory representation of variant (semi-structured) data.
///
/// Wraps a single child array that contains the actual variant-encoded data
/// (e.g. a `ParquetVariantArray` or any other variant encoding).
///
/// Nullability is delegated to the child array: `VariantArray`'s dtype is
/// always the child's dtype. The child's validity determines which rows are
/// null.
#[derive(Clone, Debug)]
pub struct VariantArray {
    child: ArrayRef,
}

impl VariantArray {
    /// Creates a new VariantArray. Nullability comes from the child's dtype.
    pub fn new(child: ArrayRef) -> Self {
        Self { child }
    }

    /// Returns a reference to the underlying child array.
    pub fn child(&self) -> &ArrayRef {
        &self.child
    }
}
