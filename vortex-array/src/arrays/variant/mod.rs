// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod vtable;

use vortex_error::VortexExpect;

pub use self::vtable::Variant;
use crate::ArrayRef;

pub(super) const NUM_SLOTS: usize = 1;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["child"];

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
    slots: [Option<ArrayRef>; NUM_SLOTS],
}

impl VariantArray {
    /// Creates a new VariantArray. Nullability comes from the child's dtype.
    pub fn new(child: ArrayRef) -> Self {
        Self {
            slots: [Some(child)],
        }
    }

    /// Returns a reference to the underlying child array.
    pub fn child(&self) -> &ArrayRef {
        self.slots[0]
            .as_ref()
            .vortex_expect("VariantArray child slot")
    }
}
