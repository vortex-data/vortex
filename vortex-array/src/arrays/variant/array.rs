// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Deref;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use super::Variant;
use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::Precision;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::TypedArrayRef;
use crate::arrays::Dict;
use crate::arrays::Filter;
use crate::arrays::Masked;
use crate::arrays::Slice;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::filter::FilterArrayExt;
use crate::arrays::masked::MaskedArrayExt;
use crate::arrays::masked::MaskedArraySlotsExt;
use crate::arrays::slice::SliceArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;

pub(crate) const CORE_STORAGE_SLOT: usize = 0;
pub(crate) const SHREDDED_SLOT: usize = 1;
pub(crate) const NUM_SLOTS: usize = 2;
pub(crate) const SLOT_NAMES: [&str; NUM_SLOTS] = ["core_storage", "shredded"];

/// Per-array metadata for [`crate::arrays::variant::VariantArray`].
///
/// The metadata records whether the optional logical `shredded` child is absent, stored inline,
/// or delegated to a child owned by an encoding inside `core_storage`.
///
/// Delegated children are identified by both the owning encoding ID and a slot name local to that
/// encoding. Slot names are not globally unique across Vortex encodings, so the slot name is never
/// interpreted until derived lookup reaches an array with the recorded encoding ID. Lookup stops
/// at that source encoding even if the slot is absent, instead of searching deeper for another
/// array with the same encoding ID. Derived lookup may pass through row-preserving wrappers around
/// `core_storage` and nested canonical [`crate::arrays::variant::VariantArray`] boundaries
/// produced by execution or normalization.
#[derive(Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum VariantMetadata {
    /// The canonical variant exposes no `shredded` child.
    #[default]
    None,
    /// The canonical variant stores its `shredded` child as physical slot 1.
    Inline { shredded_dtype: DType },
    /// The canonical variant delegates its `shredded` child to a local slot of an encoding found
    /// inside `core_storage`.
    Derived {
        /// Encoding whose local slot namespace owns `slot_name`.
        source_encoding_id: ArrayId,
        /// Slot name local to `source_encoding_id`.
        slot_name: String,
    },
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
            Self::Derived {
                source_encoding_id,
                slot_name,
            } => write!(f, "derived_shredded_slot: {source_encoding_id}.{slot_name}"),
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

    pub(crate) fn derived(source_encoding_id: ArrayId, slot_name: impl AsRef<str>) -> Self {
        Self::Derived {
            source_encoding_id,
            slot_name: slot_name.as_ref().to_string(),
        }
    }

    pub(crate) fn derived_slot(&self) -> Option<(ArrayId, &str)> {
        match self {
            Self::Derived {
                source_encoding_id,
                slot_name,
            } => Some((*source_encoding_id, slot_name.as_str())),
            Self::None | Self::Inline { .. } => None,
        }
    }

    pub(crate) fn is_derived(&self) -> bool {
        matches!(self, Self::Derived { .. })
    }
}

pub(crate) fn try_derived_shredded_from_core_storage(
    core_storage: &ArrayRef,
    source_encoding_id: ArrayId,
    slot_name: &str,
) -> VortexResult<Option<ArrayRef>> {
    // Row-preserving wrappers are transparent for derived lookup. Their local slot names, such as
    // `child`, `validity`, or `values`, must not shadow the source encoding's slot namespace.
    if let Some(slice) = core_storage.as_opt::<Slice>() {
        return try_derived_shredded_from_core_storage(
            slice.child(),
            source_encoding_id,
            slot_name,
        )?
        .map(|child| child.slice(slice.deref().slice_range().clone()))
        .transpose();
    }

    if let Some(filter) = core_storage.as_opt::<Filter>() {
        return try_derived_shredded_from_core_storage(
            filter.child(),
            source_encoding_id,
            slot_name,
        )?
        .map(|child| child.filter(filter.deref().filter_mask().clone()))
        .transpose();
    }

    if let Some(masked) = core_storage.as_opt::<Masked>() {
        return try_derived_shredded_from_core_storage(
            masked.child(),
            source_encoding_id,
            slot_name,
        )?
        .map(|child| {
            let len = child.len();
            child.mask(masked.masked_validity().to_array(len))
        })
        .transpose();
    }

    if let Some(dict) = core_storage.as_opt::<Dict>() {
        return try_derived_shredded_from_core_storage(
            dict.values(),
            source_encoding_id,
            slot_name,
        )?
        .map(|child| child.take(dict.codes().clone()))
        .transpose();
    }

    if core_storage.encoding_id() == source_encoding_id {
        for (idx, slot) in core_storage.slots().iter().enumerate() {
            if core_storage.slot_name(idx) == slot_name {
                return Ok(slot.clone());
            }
        }

        return Ok(None);
    }

    if let Some(variant) = core_storage.as_opt::<Variant>() {
        return try_derived_shredded_from_core_storage(
            variant.core_storage(),
            source_encoding_id,
            slot_name,
        );
    }

    Ok(None)
}

/// Returns the delegated `shredded` child exposed by `core_storage`, if any.
///
/// Derived children may be reconstructed through nested canonical [`Variant`] wrappers in
/// addition to row-preserving wrappers around the underlying core storage.
pub(crate) fn derived_shredded_from_core_storage(
    core_storage: &ArrayRef,
    source_encoding_id: ArrayId,
    slot_name: &str,
) -> Option<ArrayRef> {
    try_derived_shredded_from_core_storage(core_storage, source_encoding_id, slot_name)
        .vortex_expect("validated derived VariantArray shredded child")
}

/// Rebuilds a canonical [`crate::arrays::variant::VariantArray`] after transforming its
/// `core_storage`, preserving whether the logical `shredded` child is inline or derived.
///
/// If the transformed `core_storage` is itself a canonical [`Variant`] and the old derived source
/// is no longer reachable, the rebuilt array adopts that nested variant's exposed `shredded`
/// source. This keeps execution/normalization from retaining a stale source encoding ID.
pub(crate) fn rebuild_variant_array<T, F>(
    variant: &T,
    core_storage: ArrayRef,
    inline_shredded: F,
) -> VortexResult<Array<Variant>>
where
    T: VariantArrayExt + ?Sized,
    F: FnOnce() -> VortexResult<Option<ArrayRef>>,
{
    if let Some((source_encoding_id, slot_name)) = variant.derived_shredded_source() {
        let nested_variant_source = core_storage.as_opt::<Variant>().and_then(|core_variant| {
            if let Some((source_encoding_id, slot_name)) = core_variant.derived_shredded_source() {
                Some((source_encoding_id, slot_name.to_string()))
            } else if matches!(core_variant.deref(), VariantMetadata::Inline { .. }) {
                Some((
                    core_storage.encoding_id(),
                    SLOT_NAMES[SHREDDED_SLOT].to_string(),
                ))
            } else {
                None
            }
        });
        if try_derived_shredded_from_core_storage(&core_storage, source_encoding_id, slot_name)?
            .is_none()
            && let Some((core_source_encoding_id, core_slot_name)) = nested_variant_source
        {
            return Array::<Variant>::try_new_derived(
                core_storage,
                core_source_encoding_id,
                &core_slot_name,
            );
        }
        Array::<Variant>::try_new_derived(core_storage, source_encoding_id, slot_name)
    } else {
        Array::<Variant>::try_new(core_storage, inline_shredded()?)
    }
}

/// Rebuilds a canonical [`crate::arrays::variant::VariantArray`] after its physical slots were
/// transformed.
///
/// This exists because derived `shredded` metadata stores an encoding-qualified reference into
/// `core_storage`, not a physical child. If a transformation rewrites slot 0 into a nested
/// canonical [`Variant`] whose own derived source changed, the outer metadata must be rewritten to
/// keep pointing at the exposed source instead of retaining a stale encoding ID.
pub(crate) fn rebuild_variant_array_from_slots<T>(
    variant: &T,
    slots: Vec<Option<ArrayRef>>,
) -> VortexResult<Array<Variant>>
where
    T: VariantArrayExt + ?Sized,
{
    vortex_ensure!(
        slots.len() == NUM_SLOTS,
        "VariantArray expects {NUM_SLOTS} slots, got {}",
        slots.len()
    );
    let core_storage = slots[CORE_STORAGE_SLOT].clone().ok_or_else(|| {
        vortex_error::vortex_err!("VariantArray core_storage slot must be present")
    })?;
    let inline_shredded = slots[SHREDDED_SLOT].clone();

    rebuild_variant_array(variant, core_storage, || Ok(inline_shredded))
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
    /// are delegated to an encoding-qualified local slot inside [`Self::core_storage`], including
    /// through row-preserving wrappers and nested canonical [`Variant`] wrappers around that
    /// child relationship.
    fn shredded(&self) -> Option<ArrayRef> {
        match self.deref() {
            VariantMetadata::None => None,
            VariantMetadata::Inline { .. } => self.as_ref().slots()[SHREDDED_SLOT].clone(),
            VariantMetadata::Derived {
                source_encoding_id,
                slot_name,
            } => derived_shredded_from_core_storage(
                self.core_storage(),
                *source_encoding_id,
                slot_name,
            ),
        }
    }

    /// Returns the encoding-qualified delegated slot when `shredded` is derived from
    /// `core_storage`.
    ///
    /// The returned slot name is local to the returned source encoding ID. It must not be treated
    /// as a globally unique child name.
    fn derived_shredded_source(&self) -> Option<(ArrayId, &str)> {
        self.deref().derived_slot()
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
    /// child is delegated to a local slot owned by an encoding found in `core_storage`.
    ///
    /// Derived `shredded` children are accessor-only and are not stored in physical slot 1.
    /// `source_encoding_id` identifies the encoding whose local child namespace owns `slot_name`.
    /// The slot name is only interpreted after lookup reaches an array with that encoding ID,
    /// because Vortex slot names are local to each encoding and are not globally unique. Lookup
    /// stops at the first non-transparent array with that encoding ID.
    ///
    /// The delegated child is recovered from the logical `core_storage` value, including through
    /// row-preserving wrappers and nested canonical [`Variant`] wrappers that preserve the same
    /// child relationship.
    pub fn try_new_derived(
        core_storage: ArrayRef,
        source_encoding_id: ArrayId,
        slot_name: impl AsRef<str>,
    ) -> VortexResult<Self> {
        let dtype = DType::Variant(core_storage.dtype().nullability());
        let len = core_storage.len();
        let stats = core_storage.statistics().to_owned();
        let data = VariantMetadata::derived(source_encoding_id, slot_name);
        Array::try_from_parts(
            ArrayParts::new(Variant, dtype, len, data).with_slots(vec![Some(core_storage), None]),
        )
        .map(|array| array.with_stats_set(stats))
    }
}
