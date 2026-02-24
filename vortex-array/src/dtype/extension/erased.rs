// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Type-erased extension dtype ([`ExtDTypeRef`]).

use std::any::type_name;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::dtype::extension::Matcher;
use crate::dtype::extension::typed::DynExtDType;
use crate::dtype::extension::typed::ExtDTypeInner;

/// A type-erased extension dtype.
///
/// This stores an [`ExtVTable`], some type metadata, and a storage [`DType`] behind a trait object,
/// allowing heterogeneous storage inside [`DType::Extension`] (so that we do not need a generic
/// parameter).
///
/// You can use [`try_downcast()`] or [`downcast()`] to recover the concrete vtable type as an
/// [`ExtDType<V>`] (as long as you know what `V` is).
///
/// [`try_downcast()`]: ExtDTypeRef::try_downcast
/// [`downcast()`]: ExtDTypeRef::downcast
#[derive(Clone)]
pub struct ExtDTypeRef(pub(super) Arc<dyn DynExtDType>);

impl ExtDTypeRef {
    /// Returns the [`ExtId`] identifying this extension type.
    pub fn id(&self) -> ExtId {
        self.0.id()
    }

    /// Returns the storage dtype of the extension type.
    pub fn storage_dtype(&self) -> &DType {
        self.0.storage_dtype()
    }

    /// Returns the nullability of the storage dtype.
    pub fn nullability(&self) -> Nullability {
        self.storage_dtype().nullability()
    }

    /// Returns true if the storage dtype is nullable.
    pub fn is_nullable(&self) -> bool {
        self.nullability().is_nullable()
    }

    /// Returns a new [`ExtDTypeRef`] with the given nullability.
    pub fn with_nullability(&self, nullability: Nullability) -> Self {
        if self.nullability() == nullability {
            self.clone()
        } else {
            self.0.with_nullability(nullability)
        }
    }

    /// Serialize the metadata into a byte vector.
    pub fn serialize_metadata(&self) -> VortexResult<Vec<u8>> {
        self.0.metadata_serialize()
    }

    /// Returns a `Display`-able view of just the metadata.
    pub fn display_metadata(&self) -> impl Display + '_ {
        MetadataDisplay(&*self.0)
    }

    /// Compute equality ignoring nullability.
    pub fn eq_ignore_nullability(&self, other: &Self) -> bool {
        self.id() == other.id()
            && self.0.metadata_eq(other.0.metadata_any())
            && self
                .storage_dtype()
                .eq_ignore_nullability(other.storage_dtype())
    }
}

/// Methods for downcasting type-erased extension dtypes.
impl ExtDTypeRef {
    /// Check if the extension dtype is of the concrete type.
    pub fn is<M: Matcher>(&self) -> bool {
        M::matches(self)
    }

    /// Extract the metadata of the ExtDType per the given [`Matcher`].
    pub fn metadata_opt<M: Matcher>(&self) -> Option<M::Match<'_>> {
        M::try_match(self)
    }

    /// Extract the metadata of the [`ExtDType`] per the given [`Matcher`].
    ///
    /// # Panics
    ///
    /// Panics if the match fails.
    pub fn metadata<M: Matcher>(&self) -> M::Match<'_> {
        self.metadata_opt::<M>()
            .vortex_expect("Failed to downcast ExtDTypeRef")
    }

    /// Downcast to the concrete [`ExtDType`].
    ///
    /// Returns `Err(self)` if the downcast fails.
    pub fn try_downcast<V: ExtVTable>(self) -> Result<ExtDType<V>, ExtDTypeRef> {
        if self.0.as_any().is::<ExtDTypeInner<V>>() {
            // SAFETY: type matches and ExtDTypeInner<V> is the only implementor
            let ptr = Arc::into_raw(self.0) as *const ExtDTypeInner<V>;
            let inner = unsafe { Arc::from_raw(ptr) };
            Ok(ExtDType(inner))
        } else {
            Err(self)
        }
    }

    /// Downcast to the concrete [`ExtDType`].
    ///
    /// # Panics
    ///
    /// Panics if the downcast fails.
    pub fn downcast<V: ExtVTable>(self) -> ExtDType<V> {
        self.try_downcast::<V>()
            .map_err(|this| {
                vortex_err!(
                    "Failed to downcast ExtDTypeRef {} to {}",
                    this.0.id(),
                    type_name::<V>(),
                )
            })
            .vortex_expect("Failed to downcast ExtDTypeRef")
    }
}

impl Display for ExtDTypeRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let metadata = self.display_metadata().to_string();
        if metadata.is_empty() {
            write!(f, "{}", self.id())?;
        } else {
            write!(f, "{}[{}]", self.id(), metadata)?;
        }
        write!(f, "({})", self.storage_dtype())
    }
}

impl Debug for ExtDTypeRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtDType")
            .field("id", &self.id())
            .field("metadata", &MetadataDebug(&*self.0))
            .field("storage_dtype", &self.storage_dtype())
            .finish()
    }
}

impl PartialEq for ExtDTypeRef {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
            && self.0.metadata_eq(other.0.metadata_any())
            && self.storage_dtype() == other.storage_dtype()
    }
}
impl Eq for ExtDTypeRef {}

impl Hash for ExtDTypeRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id().hash(state);
        self.0.metadata_hash(state);
        self.storage_dtype().hash(state);
    }
}

// Private formatting helpers for Display and Debug impls.

struct MetadataDisplay<'a>(&'a dyn DynExtDType);

impl Display for MetadataDisplay<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.metadata_display(f)
    }
}

struct MetadataDebug<'a>(&'a dyn DynExtDType);

impl Debug for MetadataDebug<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.metadata_debug(f)
    }
}
