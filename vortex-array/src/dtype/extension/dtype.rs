// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Type-erased and typed extension data types.
//!
//! Extension dtypes wrap a storage [`DType`] together with an [`ExtVTable`] that gives the dtype
//! semantic meaning beyond its raw storage representation.
//!
//! There are two main public types:
//!
//! - [`ExtDTypeRef`]: A type-erased extension dtype that can be stored heterogeneously.
//! - [`ExtDType`]: A typed extension dtype parameterized by a concrete [`ExtVTable`]
//!   implementation.

use std::any::Any;
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
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::dtype::extension::Matcher;

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
pub struct ExtDTypeRef(pub(super) Arc<dyn ExtDTypeImpl>);

/// A typed extension data type, parameterized by a concrete [`ExtVTable`].
///
/// You can construct one of these via [`try_new()`] (for zero-sized vtables) or
/// [`try_with_vtable()`], and erase the type with [`erased()`] to obtain an [`ExtDTypeRef`].
///
/// [`try_new()`]: ExtDType::try_new
/// [`try_with_vtable()`]: ExtDType::try_with_vtable
/// [`erased()`]: ExtDType::erased
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExtDType<V: ExtVTable>(Arc<ExtDTypeAdapter<V>>);

/// The concrete inner representation of an extension dtype, pairing a vtable with its metadata
/// and storage dtype.
///
/// This is the sole implementor of [`ExtDTypeImpl`], enabling [`ExtDTypeRef`] to safely downcast
/// back to the concrete vtable type via [`Any`].
#[derive(Debug, PartialEq, Eq, Hash)]
pub(super) struct ExtDTypeAdapter<V: ExtVTable> {
    /// The extension dtype vtable.
    vtable: V,
    /// The extension dtype metadata.
    pub(super) metadata: V::Metadata,
    /// The underlying storage dtype.
    storage_dtype: DType,
}

impl<V: ExtVTable> ExtDTypeImpl for ExtDTypeAdapter<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn id(&self) -> ExtId {
        self.vtable.id()
    }

    fn storage_dtype(&self) -> &DType {
        &self.storage_dtype
    }

    fn metadata_any(&self) -> &dyn Any {
        &self.metadata
    }

    fn metadata_debug(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        <V::Metadata as Debug>::fmt(&self.metadata, f)
    }

    fn metadata_display(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        <V::Metadata as Display>::fmt(&self.metadata, f)
    }

    fn metadata_eq(&self, other: &dyn Any) -> bool {
        let Some(other) = other.downcast_ref::<V::Metadata>() else {
            return false;
        };
        <V::Metadata as PartialEq>::eq(&self.metadata, other)
    }

    fn metadata_hash(&self, mut state: &mut dyn Hasher) {
        <V::Metadata as Hash>::hash(&self.metadata, &mut state);
    }

    fn metadata_serialize(&self) -> VortexResult<Vec<u8>> {
        V::serialize(&self.vtable, &self.metadata)
    }

    fn with_nullability(&self, nullability: Nullability) -> ExtDTypeRef {
        let storage_dtype = self.storage_dtype.with_nullability(nullability);
        ExtDType::<V>::try_with_vtable(self.vtable.clone(), self.metadata.clone(), storage_dtype)
            .vortex_expect(
                "Extension DType should not fail validation with the same storage type \
                 but different nullability",
            )
            .erased()
    }
}

/// Convenience implementation for zero-sized VTables (or VTables that implement `Default`).
impl<V: ExtVTable + Default> ExtDType<V> {
    /// Creates a new extension dtype with the given metadata and storage dtype.
    pub fn try_new(metadata: V::Metadata, storage_dtype: DType) -> VortexResult<Self> {
        Self::try_with_vtable(V::default(), metadata, storage_dtype)
    }
}

impl<V: ExtVTable> ExtDType<V> {
    /// Creates a new extension dtype with the given metadata and storage dtype.
    pub fn try_with_vtable(
        vtable: V,
        metadata: V::Metadata,
        storage_dtype: DType,
    ) -> VortexResult<Self> {
        vtable.validate_dtype(&metadata, &storage_dtype)?;

        Ok(Self(Arc::new(ExtDTypeAdapter::<V> {
            vtable,
            metadata,
            storage_dtype,
        })))
    }

    /// Returns the identifier of the extension type.
    pub fn id(&self) -> ExtId {
        self.0.id()
    }

    /// Returns the vtable of the extension type.
    pub fn vtable(&self) -> &V {
        &self.0.vtable
    }

    /// Returns the metadata of the extension type.
    pub fn metadata(&self) -> &V::Metadata {
        &self.0.metadata
    }

    /// Returns the storage dtype of the extension type.
    pub fn storage_dtype(&self) -> &DType {
        &self.0.storage_dtype
    }

    /// Erase the concrete type information, returning a type-erased extension dtype.
    pub fn erased(self) -> ExtDTypeRef {
        ExtDTypeRef(self.0)
    }
}

// NB: If you need access to the vtable, you probably want to add a method and implementation to
// `ExtDTypeImpl` and `ExtDTypeAdapter`.
impl ExtDTypeRef {
    /// Returns the [`ExtId`] identifying this extension type.
    pub fn id(&self) -> ExtId {
        self.0.id()
    }

    /// Returns the type-erased metadata of the extension type.
    pub fn metadata_erased(&self) -> ExtDTypeMetadata<'_> {
        ExtDTypeMetadata { ext_dtype: self }
    }

    /// Returns the storage dtype of the extension type.
    pub fn storage_dtype(&self) -> &DType {
        self.0.storage_dtype()
    }

    /// Returns the nullability of the storage dtype.
    pub fn nullability(&self) -> Nullability {
        // The nullability of the extension type must always be the same as the storage type.
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

    /// Compute equality ignoring nullability.
    pub fn eq_ignore_nullability(&self, other: &Self) -> bool {
        self.id() == other.id()
            && self.metadata_erased() == other.metadata_erased()
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
        // Check if the concrete type matches
        if self.0.as_any().is::<ExtDTypeAdapter<V>>() {
            // SAFETY: type matches and ExtDTypeImpl<V> is the only implementor
            let ptr = Arc::into_raw(self.0) as *const ExtDTypeAdapter<V>;
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
        let metadata = format!("{}", self.metadata_erased());
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
            .field("metadata", &self.metadata_erased())
            .field("storage_dtype", &self.storage_dtype())
            .finish()
    }
}

impl PartialEq for ExtDTypeRef {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
            && self.metadata_erased() == other.metadata_erased()
            && self.storage_dtype() == other.storage_dtype()
    }
}
impl Eq for ExtDTypeRef {}

impl Hash for ExtDTypeRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id().hash(state);
        self.metadata_erased().hash(state);
        self.storage_dtype().hash(state);
    }
}

/// A wrapper providing type-erased access to extension dtype metadata.
///
/// This delegates [`Display`], [`Debug`], [`PartialEq`], [`Hash`], and serialization to the
/// underlying `ExtDTypeImpl` trait object, allowing callers to work with metadata without
/// knowing the concrete vtable type.
pub struct ExtDTypeMetadata<'a> {
    ext_dtype: &'a ExtDTypeRef,
}

impl ExtDTypeMetadata<'_> {
    /// Serialize the metadata into a byte vector.
    pub fn serialize(&self) -> VortexResult<Vec<u8>> {
        self.ext_dtype.0.metadata_serialize()
    }
}

impl Display for ExtDTypeMetadata<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.ext_dtype.0.metadata_display(f)
    }
}

impl Debug for ExtDTypeMetadata<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.ext_dtype.0.metadata_debug(f)
    }
}

impl PartialEq for ExtDTypeMetadata<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.ext_dtype
            .0
            .metadata_eq(other.ext_dtype.0.metadata_any())
    }
}
impl Eq for ExtDTypeMetadata<'_> {}

impl Hash for ExtDTypeMetadata<'_> {
    fn hash<H: Hasher>(&self, mut state: &mut H) {
        self.ext_dtype.0.metadata_hash(&mut state);
    }
}

/// An object-safe, sealed trait encapsulating the behavior for extension dtypes.
///
/// This mirrors `DynExtScalarValue` in `vortex-scalar`: it provides type-erased access to the
/// extension dtype's identity, storage dtype, and metadata. The only implementor is
/// [`ExtDTypeAdapter`].
pub(super) trait ExtDTypeImpl: 'static + Send + Sync + super::sealed::Sealed {
    /// Returns `self` as a trait object for downcasting.
    fn as_any(&self) -> &dyn Any;
    /// Returns the [`ExtId`] identifying this extension type.
    fn id(&self) -> ExtId;
    /// Returns a reference to the storage [`DType`].
    fn storage_dtype(&self) -> &DType;
    /// Returns the metadata as a trait object for downcasting.
    fn metadata_any(&self) -> &dyn Any;
    /// Formats the metadata using [`Debug`].
    fn metadata_debug(&self, f: &mut Formatter<'_>) -> std::fmt::Result;
    /// Formats the metadata using [`Display`].
    fn metadata_display(&self, f: &mut Formatter<'_>) -> std::fmt::Result;
    /// Checks equality of the metadata against a type-erased value.
    fn metadata_eq(&self, other: &dyn Any) -> bool;
    /// Hashes the metadata into the given [`Hasher`].
    fn metadata_hash(&self, state: &mut dyn Hasher);
    /// Serializes the metadata into a byte vector.
    fn metadata_serialize(&self) -> VortexResult<Vec<u8>>;
    /// Returns a new [`ExtDTypeRef`] with the given nullability.
    fn with_nullability(&self, nullability: Nullability) -> ExtDTypeRef;
}
