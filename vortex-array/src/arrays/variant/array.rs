// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Deref;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::Variant;
use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::Precision;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::IntoArray;
use crate::array::TypedArrayRef;
use crate::arrays::Dict;
use crate::arrays::Filter;
use crate::arrays::Masked;
use crate::arrays::MaskedArray;
use crate::arrays::Slice;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::filter::FilterArrayExt;
use crate::arrays::masked::MaskedArrayExt;
use crate::arrays::masked::MaskedArraySlotsExt;
use crate::arrays::slice::SliceArrayExt;
use crate::dtype::DType;

pub(crate) const CORE_STORAGE_SLOT: usize = 0;
pub(crate) const SHREDDED_SLOT: usize = 1;
pub(crate) const NUM_SLOTS: usize = 2;
pub(crate) const SLOT_NAMES: [&str; NUM_SLOTS] = ["core_storage", "shredded"];

/// Per-array metadata for [`crate::arrays::variant::VariantArray`].
///
/// The metadata records whether the optional logical `shredded` child is absent, stored inline,
/// or delegated to a named child of `core_storage`. Delegated children are defined logically, so
/// they may be recovered through row-preserving wrappers around `core_storage` and nested
/// canonical [`crate::arrays::variant::VariantArray`] boundaries produced by execution or
/// normalization.
#[derive(Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum VariantMetadata {
    /// The canonical variant exposes no `shredded` child.
    #[default]
    None,
    /// The canonical variant stores its `shredded` child as physical slot 1.
    Inline { shredded_dtype: DType },
    /// The canonical variant delegates its `shredded` child to a named child of `core_storage`.
    Derived { slot_name: String },
}

impl ArrayHash for VariantMetadata {
    fn array_hash<H: Hasher>(&self, state: &mut H, _precision: Precision) {
        self.hash(state);
    }
}

impl ArrayEq for VariantMetadata {
    fn array_eq(&self, other: &Self, _precision: Precision) -> bool {
        self == other
    }
}

impl Display for VariantMetadata {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => Ok(()),
            Self::Inline { shredded_dtype } => write!(f, "inline_shredded_dtype: {shredded_dtype}"),
            Self::Derived { slot_name } => write!(f, "derived_shredded_slot: {slot_name}"),
        }
    }
}

impl VariantMetadata {
    pub(crate) fn from_inline_shredded(shredded: Option<&ArrayRef>) -> Self {
        match shredded {
            Some(shredded) => Self::Inline {
                shredded_dtype: shredded.dtype().clone(),
            },
            None => Self::None,
        }
    }

    pub(crate) fn derived(slot_name: impl AsRef<str>) -> Self {
        Self::Derived {
            slot_name: slot_name.as_ref().to_string(),
        }
    }

    pub(crate) fn derived_slot_name(&self) -> Option<&str> {
        match self {
            Self::Derived { slot_name } => Some(slot_name.as_str()),
            Self::None | Self::Inline { .. } => None,
        }
    }

    pub(crate) fn is_derived(&self) -> bool {
        matches!(self, Self::Derived { .. })
    }
}

pub(crate) fn try_derived_shredded_from_core_storage(
    core_storage: &ArrayRef,
    slot_name: &str,
) -> VortexResult<Option<ArrayRef>> {
    if let Some(slot) = core_storage
        .slots()
        .iter()
        .enumerate()
        .find_map(|(idx, slot)| {
            (core_storage.slot_name(idx) == slot_name)
                .then(|| slot.clone())
                .flatten()
        })
    {
        return Ok(Some(slot));
    }

    if let Some(variant) = core_storage.as_opt::<Variant>() {
        return try_derived_shredded_from_core_storage(variant.core_storage(), slot_name);
    }

    if let Some(slice) = core_storage.as_opt::<Slice>() {
        return try_derived_shredded_from_core_storage(slice.child(), slot_name)?
            .map(|child| child.slice(slice.deref().slice_range().clone()))
            .transpose();
    }

    if let Some(filter) = core_storage.as_opt::<Filter>() {
        return try_derived_shredded_from_core_storage(filter.child(), slot_name)?
            .map(|child| child.filter(filter.deref().filter_mask().clone()))
            .transpose();
    }

    if let Some(masked) = core_storage.as_opt::<Masked>() {
        return try_derived_shredded_from_core_storage(masked.child(), slot_name)?
            .map(|child| {
                MaskedArray::try_new(child, masked.masked_validity())
                    .map(|masked| masked.into_array())
            })
            .transpose();
    }

    if let Some(dict) = core_storage.as_opt::<Dict>() {
        return try_derived_shredded_from_core_storage(dict.values(), slot_name)?
            .map(|child| child.take(dict.codes().clone()))
            .transpose();
    }

    Ok(None)
}

/// Returns the delegated `shredded` child exposed by `core_storage`, if any.
///
/// Derived children may be reconstructed through nested canonical [`Variant`] wrappers in
/// addition to row-preserving wrappers around the underlying core storage.
pub(crate) fn derived_shredded_from_core_storage(
    core_storage: &ArrayRef,
    slot_name: &str,
) -> Option<ArrayRef> {
    try_derived_shredded_from_core_storage(core_storage, slot_name)
        .vortex_expect("validated derived VariantArray shredded child")
}

/// Rebuilds a canonical [`crate::arrays::variant::VariantArray`] after transforming its
/// `core_storage`, preserving whether the logical `shredded` child is inline or derived.
pub(crate) fn rebuild_variant_array<T, F>(
    variant: &T,
    core_storage: ArrayRef,
    inline_shredded: F,
) -> VortexResult<Array<Variant>>
where
    T: VariantArrayExt + ?Sized,
    F: FnOnce() -> VortexResult<Option<ArrayRef>>,
{
    if let Some(slot_name) = variant.derived_shredded_slot_name() {
        Array::<Variant>::try_new_derived(core_storage, slot_name)
    } else {
        Array::<Variant>::try_new(core_storage, inline_shredded()?)
    }
}

/// Accessors for the canonical children of a [`crate::arrays::variant::VariantArray`].
pub trait VariantArrayExt: TypedArrayRef<Variant> {
    /// Returns the mandatory child that owns the variant encoding semantics.
    ///
    /// The outer variant dtype, length, validity, and scalar semantics are all defined by this
    /// child.
    fn core_storage(&self) -> &ArrayRef {
        self.as_ref().slots()[CORE_STORAGE_SLOT]
            .as_ref()
            .vortex_expect("validated variant core storage slot")
    }

    /// Returns the optional logical `shredded` child, if one is present.
    ///
    /// Inline `shredded` children are returned from physical slot 1. Derived `shredded` children
    /// are delegated to a named child of [`Self::core_storage`], including through row-preserving
    /// wrappers and nested canonical [`Variant`] wrappers around that child.
    fn shredded(&self) -> Option<ArrayRef> {
        match self.deref() {
            VariantMetadata::None => None,
            VariantMetadata::Inline { .. } => self.as_ref().slots()[SHREDDED_SLOT].clone(),
            VariantMetadata::Derived { slot_name } => {
                derived_shredded_from_core_storage(self.core_storage(), slot_name)
            }
        }
    }

    /// Returns the delegated slot name when `shredded` is derived from `core_storage`.
    fn derived_shredded_slot_name(&self) -> Option<&str> {
        self.deref().derived_slot_name()
    }

    /// Returns whether the logical `shredded` child is derived from `core_storage`.
    fn shredded_is_derived(&self) -> bool {
        self.deref().is_derived()
    }
}
impl<T: TypedArrayRef<Variant>> VariantArrayExt for T {}

impl Array<Variant> {
    /// Creates a canonical [`crate::arrays::variant::VariantArray`] from a required
    /// `core_storage` child and an optional inline `shredded` child.
    ///
    /// `core_storage` defines the outer variant [`crate::dtype::DType`], array length, and
    /// statistics. When `shredded` is present it must have the same length as `core_storage` and
    /// is serialized as physical slot 1.
    pub fn try_new(core_storage: ArrayRef, shredded: Option<ArrayRef>) -> VortexResult<Self> {
        let dtype = DType::Variant(core_storage.dtype().nullability());
        let len = core_storage.len();
        let stats = core_storage.statistics().to_owned();
        let data = VariantMetadata::from_inline_shredded(shredded.as_ref());
        Array::try_from_parts(
            ArrayParts::new(Variant, dtype, len, data)
                .with_slots(vec![Some(core_storage), shredded]),
        )
        .map(|array| array.with_stats_set(stats))
    }

    /// Creates a canonical [`crate::arrays::variant::VariantArray`] whose logical `shredded`
    /// child is delegated to a named child of `core_storage`.
    ///
    /// Derived `shredded` children are accessor-only and are not stored in physical slot 1. The
    /// delegated child is recovered from the logical `core_storage` value, including through
    /// row-preserving wrappers and nested canonical [`Variant`] wrappers that preserve the same
    /// child relationship.
    pub fn try_new_derived(
        core_storage: ArrayRef,
        slot_name: impl AsRef<str>,
    ) -> VortexResult<Self> {
        let dtype = DType::Variant(core_storage.dtype().nullability());
        let len = core_storage.len();
        let stats = core_storage.statistics().to_owned();
        let data = VariantMetadata::derived(slot_name);
        Array::try_from_parts(
            ArrayParts::new(Variant, dtype, len, data).with_slots(vec![Some(core_storage), None]),
        )
        .map(|array| array.with_stats_set(stats))
    }
}
