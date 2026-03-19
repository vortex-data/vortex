// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod vtable;

pub use self::vtable::Variant;
use crate::ArrayRef;
use crate::dtype::DType;
use crate::dtype::Nullability;
use vortex_error::VortexExpect;

pub(super) const NUM_SLOTS: usize = 1;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["child"];

/// The canonical in-memory representation of variant (semi-structured) data.
///
/// Wraps a single child array that contains the actual variant-encoded data
/// (e.g. a `ParquetVariantArray` or any other variant encoding).
#[derive(Clone, Debug)]
pub struct VariantArray {
    dtype: DType,
    slots: [Option<ArrayRef>; NUM_SLOTS],
}

impl VariantArray {
    /// Creates a new VariantArray with the given nullability.
    pub fn new(child: ArrayRef, nullability: Nullability) -> Self {
        Self {
            dtype: DType::Variant(nullability),
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
