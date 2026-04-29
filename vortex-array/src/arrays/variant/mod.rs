// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Canonical [`VariantArray`] layout.
//!
//! A canonical variant array has two logical children:
//! - slot 0 is the mandatory `core_storage` child
//! - slot 1 is the optional `shredded` child
//!
//! `core_storage` owns the encoding-specific semantics of the variant values. The outer variant
//! [`DType`] and array length are derived from it, and scalar extraction and
//! validity delegate to it.
//!
//! `shredded` is a same-length auxiliary child that exposes a more concrete structure when one is
//! available. It may either be stored inline or be derived from an encoding-qualified child inside
//! `core_storage`.
//!
//! Canonical variants always keep two physical slots: slot 0 stores `core_storage`, and slot 1 is
//! used only for inline `shredded`. Derived `shredded` children are accessor-only and are
//! delegated to a slot name owned by a specific source encoding. Slot names are local to each
//! encoding and must not be treated as globally unique.
//!
//! That delegation is defined over the logical `core_storage` value, not just its top-level
//! slots. If `core_storage` is wrapped by row-preserving encodings such as slice, filter, take,
//! or masked validity, the delegated `shredded` child is reconstructed through the same wrapper.
//! The local slot name is only resolved once lookup reaches the recorded source encoding, and
//! lookup stops there if that source does not expose the slot.
//!
//! ## Canonicalization
//!
//! Recursive canonicalization preserves `core_storage` as-is. Inline `shredded` children are
//! canonicalized independently; derived `shredded` children continue to be delegated from
//! `core_storage`.
//!
//! [`DType`]: crate::dtype::DType

mod array;
#[cfg(test)]
mod tests;
mod vtable;

pub(super) use self::array::CORE_STORAGE_SLOT;
pub(super) use self::array::NUM_SLOTS;
pub(super) use self::array::SHREDDED_SLOT;
pub(super) use self::array::SLOT_NAMES;
pub use self::array::VariantArrayExt;
pub use self::array::VariantMetadata;
pub(crate) use self::array::rebuild_variant_array;
pub(crate) use self::array::rebuild_variant_array_from_slots;
pub(crate) use self::array::try_derived_shredded_from_core_storage;
pub use self::vtable::Variant;
pub use self::vtable::VariantArray;
