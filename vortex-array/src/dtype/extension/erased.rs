// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Type-erased extension dtype ([`ExtDTypeRef`]).

use std::any::type_name;
use std::fmt;
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
use crate::scalar::ScalarValue;

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

    /// Compute equality ignoring nullability.
    pub fn eq_ignore_nullability(&self, other: &Self) -> bool {
        self.id() == other.id()
            && self.0.metadata_eq(other.0.metadata_any())
            && self
                .storage_dtype()
                .eq_ignore_nullability(other.storage_dtype())
    }

    // TODO(connor): We should add a different type that returns something that can be serialized.
    /// Serialize the metadata into a byte vector.
    pub fn serialize_metadata(&self) -> VortexResult<Vec<u8>> {
        self.0.serialize_metadata()
    }

    /// Returns a `Display`-able view of just the metadata.
    pub fn display_metadata(&self) -> impl fmt::Display + '_ {
        MetadataDisplay(&*self.0)
    }

    /// Formats an extension scalar value using the current dtype for metadata context.
    pub(crate) fn fmt_storage_value<'a>(
        &'a self,
        f: &mut fmt::Formatter<'_>,
        storage_value: &'a ScalarValue,
    ) -> fmt::Result {
        self.0.value_display(f, storage_value)
    }

    /// Validates that the given storage scalar value is valid for this dtype.
    pub(crate) fn validate_storage_value(&self, storage_value: &ScalarValue) -> VortexResult<()> {
        self.0.validate_scalar_value(storage_value)
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
    pub fn try_downcast<V: ExtVTable>(self) -> Result<Arc<ExtDType<V>>, ExtDTypeRef> {
        if self.0.as_any().is::<ExtDType<V>>() {
            let ptr = Arc::into_raw(self.0) as *const ExtDType<V>;
            Ok(unsafe { Arc::from_raw(ptr) })
        } else {
            Err(self)
        }
    }

    /// Downcast to the concrete [`ExtDType`].
    ///
    /// # Panics
    ///
    /// Panics if the downcast fails.
    pub fn downcast<V: ExtVTable>(self) -> Arc<ExtDType<V>> {
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

impl fmt::Debug for ExtDTypeRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let metadata = self.0.metadata_debug(f);

        f.debug_struct("ExtDType")
            .field("id", &self.id())
            .field("metadata", &metadata)
            .field("storage_dtype", &self.storage_dtype())
            .finish()
    }
}

impl fmt::Display for ExtDTypeRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let metadata = MetadataDisplay(&*self.0).to_string();

        if metadata.is_empty() {
            write!(f, "{}", self.id())?;
        } else {
            write!(f, "{}[{}]", self.id(), metadata)?;
        }

        write!(f, "({})", self.storage_dtype())
    }
}

// Private formatting helpers for Display and Debug impls.

struct MetadataDisplay<'a>(&'a dyn DynExtDType);
impl fmt::Display for MetadataDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.metadata_display(f)
    }
}

// struct PythonDisplay<'a>(&'a dyn DynExtDType);
// impl fmt::Display for PythonDisplay<'_> {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         let metadata = MetadataDisplay(self.0).to_string();

//         let id = self.0.id();
//         let escaped_id = id.as_ref().escape_default();
//         if metadata.is_empty() {
//             write!(f, "\"{escaped_id}\"",)?;
//         } else {
//             write!(f, "\"{escaped_id}\"[{}]", metadata)?;
//         }

//         write!(f, "({})", self.0.storage_dtype())
//     }
// }
