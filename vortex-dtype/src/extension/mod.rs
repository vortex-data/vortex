// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Extension DTypes

mod matcher;
mod vtable;

use std::any::Any;
use std::any::type_name;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use arcref::ArcRef;
pub use matcher::*;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
pub use vtable::*;

use crate::DType;
use crate::Nullability;

/// A unique identifier for an extension type
pub type ExtID = ArcRef<str>;

/// An extension data type.
#[derive(Clone)]
pub struct ExtDType<V: ExtDTypeVTable>(Arc<ExtDTypeAdapter<V>>);

// Convenience impls for zero-sized VTables
impl<V: ExtDTypeVTable + Default> ExtDType<V> {
    /// Creates a new extension dtype with the given metadata and storage dtype.
    pub fn try_new(metadata: V::Metadata, storage_dtype: DType) -> VortexResult<Self> {
        Self::try_with_vtable(V::default(), metadata, storage_dtype)
    }
}

impl<V: ExtDTypeVTable> ExtDType<V> {
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
    pub fn id(&self) -> ExtID {
        self.0.id()
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

/// Type-erased extension dtype - for heterogeneous storage
#[derive(Clone)]
pub struct ExtDTypeRef(Arc<dyn ExtDTypeImpl>);

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

impl ExtDTypeRef {
    /// Returns the identifier of the extension type.
    pub fn id(&self) -> ExtID {
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

    /// Returns a new ExtDTypeRef with the given nullability.
    pub fn with_nullability(&self, nullability: Nullability) -> Self {
        if self.storage_dtype().nullability() == nullability {
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

    /// Extract the metadata of the ExtDType per the given [`Matcher`].
    ///
    /// # Panics
    ///
    /// Panics if the match fails.
    pub fn metadata<M: Matcher>(&self) -> M::Match<'_> {
        self.metadata_opt::<M>()
            .vortex_expect("Failed to downcast DynExtDType")
    }

    /// Downcast to the concrete [`ExtDType`].
    ///
    /// Returns `Err(self)` if the downcast fails.
    pub fn try_downcast<V: ExtDTypeVTable>(self) -> Result<ExtDType<V>, ExtDTypeRef> {
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
    pub fn downcast<V: ExtDTypeVTable>(self) -> ExtDType<V> {
        self.try_downcast::<V>()
            .map_err(|this| {
                vortex_err!(
                    "Failed to downcast DynExtDType {} to {}",
                    this.0.id(),
                    type_name::<V>(),
                )
            })
            .vortex_expect("Failed to downcast DynExtDType")
    }
}

/// Wrapper for type-erased extension dtype metadata.
pub struct ExtDTypeMetadata<'a> {
    pub(super) ext_dtype: &'a ExtDTypeRef,
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

/// An object-safe trait encapsulating the behavior for extension DTypes.
trait ExtDTypeImpl: 'static + Send + Sync + private::Sealed {
    fn as_any(&self) -> &dyn Any;
    fn id(&self) -> ExtID;
    fn storage_dtype(&self) -> &DType;
    fn metadata_any(&self) -> &dyn Any;
    fn metadata_debug(&self, f: &mut Formatter<'_>) -> std::fmt::Result;
    fn metadata_display(&self, f: &mut Formatter<'_>) -> std::fmt::Result;
    fn metadata_eq(&self, other: &dyn Any) -> bool;
    fn metadata_hash(&self, state: &mut dyn Hasher);
    fn metadata_serialize(&self) -> VortexResult<Vec<u8>>;
    fn with_nullability(&self, nullability: Nullability) -> ExtDTypeRef;
}

struct ExtDTypeAdapter<V: ExtDTypeVTable> {
    vtable: V,
    metadata: V::Metadata,
    storage_dtype: DType,
}

impl<V: ExtDTypeVTable> ExtDTypeImpl for ExtDTypeAdapter<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn id(&self) -> ExtID {
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
            .vortex_expect("Extension DType {} incorrect fails validation with the same storage type but different nullability").erased()
    }
}

mod private {
    use super::ExtDTypeAdapter;

    pub trait Sealed {}
    impl<V: super::ExtDTypeVTable> Sealed for ExtDTypeAdapter<V> {}
}
