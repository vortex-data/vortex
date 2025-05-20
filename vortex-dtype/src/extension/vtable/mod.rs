mod temporal;

use std::any::{Any, type_name};
use std::fmt::Debug;
use std::ops::Deref;
use std::sync::Arc;

use arcref::ArcRef;
use vortex_error::{VortexExpect, VortexResult};

use crate::datetime::TIME_ID;
use crate::{ExtID, ExtMetadata};

/// Convert a typed value into a type-erased handle to an extension type.
pub trait IntoExtensionTypeRef {
    /// Consume the receiver and return a new cloneable handle to an extension type.
    fn into_extension_type_ref(self) -> ExtensionTypeRef;
}

/// VTable for extension types.
pub trait ExtensionVTable: Send + Sync + Debug + 'static {
    /// The extension type that this VTable operates over.
    type ExtType: 'static
        + Send
        + Sync
        + Debug
        + Deref<Target = dyn ExtensionType>
        + IntoExtensionTypeRef;

    /// The encoding type that implements the serde logic for the extension type's metadata.
    type ExtEncoding: 'static + Send + Sync + Debug + Deref<Target = dyn ExtensionTypeEncoding>;

    /// Get the extension type ID.
    fn id(extension: &Self::ExtType) -> &ExtID;

    /// Serialize the metadata for an extension type instance, if there is any.
    fn serialize_metadata(extension: &Self::ExtType) -> Option<ExtMetadata>;

    /// Attempt to decode a concrete extension type instance from the ID and metadata.
    ///
    /// If this VTable doesn't support the pair, an inner `None` value is returned.
    fn try_decode(
        id: &ExtID,
        metadata: Option<&ExtMetadata>,
    ) -> VortexResult<Option<Self::ExtType>>;
}

/// A dyn-compatible trait that all extension types conform to.
///
/// This trait should never be derived directly, instead the [`ExtensionVTable`] trait should be implemented
/// by consumers and a suitable `ExtensionType` is generated via a blanket implementation.
pub trait ExtensionType: Send + Sync + Debug + private::Sealed + 'static {
    /// Entrypoint for downcasting to a concrete subtype.
    fn as_any(&self) -> &dyn Any;

    /// Get the extension type ID.
    fn id(&self) -> &ExtID;

    /// Serialize the extension type metadata out, or returns `None` if it holds none.
    fn serialize_metadata(&self) -> Option<ExtMetadata>;
}

/// A cheaply cloneable type-erased handle for extension types.
pub type ExtensionTypeRef = Arc<dyn ExtensionType>;

impl dyn ExtensionType + '_ {
    /// Predicate if the inner extension type is the one associated with the provided `VTable`.
    pub fn is<V: ExtensionVTable>(&self) -> bool {
        self.as_opt::<V>().is_some()
    }

    /// Force downcast, panicking on failure.
    ///
    /// See also: [`Self::as_opt`]
    pub fn as_<V: ExtensionVTable>(&self) -> &V::ExtType {
        self.as_opt::<V>()
            .vortex_expect("ExtensionType not of expected type")
    }

    /// Downcast to the extension type encoded in the VTable, or `None` if downcast fails.
    pub fn as_opt<V: ExtensionVTable>(&self) -> Option<&V::ExtType> {
        self.as_any()
            .downcast_ref::<ExtensionTypeAdapter<V>>()
            .map(|adapter| &adapter.0)
    }
}

mod private {
    use super::{ExtensionTypeAdapter, ExtensionVTable};

    pub trait Sealed {}

    impl<Type: ExtensionVTable> Sealed for ExtensionTypeAdapter<Type> {}
}

/// Extension type adapter for VTables.
#[derive(Debug)]
#[repr(transparent)]
pub struct ExtensionTypeAdapter<V: ExtensionVTable>(V::ExtType);

impl<V: ExtensionVTable> ExtensionType for ExtensionTypeAdapter<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn id(&self) -> &ExtID {
        V::id(&self.0)
    }

    /// Serialize a copy of the metadata out of the current extension type variant unit.
    fn serialize_metadata(&self) -> Option<ExtMetadata> {
        V::serialize_metadata(&self.0)
    }
}

/// Dyn-compatible trait for extension type serialization.
pub trait ExtensionTypeEncoding: 'static + Send + Sync + Debug {
    /// Entrypoint for downcasting to a concrete subtype.
    fn as_any(&self) -> &dyn Any;

    /// ID for the encoder. This is NOT the same thing as the ID of the extension type. This
    /// is just meant to be a unique ID for the extension loader.
    fn id(&self) -> &str;

    /// Predicate indicating if the encoding supports the given type ID.
    fn supports_type(&self, id: &ExtID) -> bool;

    /// See if this deserializes into one of the builtin extension types that are supported
    /// by this registry, propagating any deserialization errors.
    ///
    /// An inner `None` is returned if this encoding does not support serde for this extension type.
    fn try_decode(
        &self,
        ext_id: &ExtID,
        metadata: Option<&ExtMetadata>,
    ) -> VortexResult<Option<ExtensionTypeRef>>;
}

/// Implement an extension type encoding using a VTable.
#[derive(Debug)]
pub struct ExtensionTypeEncodingAdapter<V: ExtensionVTable>(V::ExtEncoding);

impl<V: ExtensionVTable> ExtensionTypeEncoding for ExtensionTypeEncodingAdapter<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn id(&self) -> &str {
        type_name::<V>()
    }

    fn supports_type(&self, id: &ExtID) -> bool {
        id.as_ref() == TIME_ID.as_ref()
    }

    fn try_decode(
        &self,
        ext_id: &ExtID,
        metadata: Option<&ExtMetadata>,
    ) -> VortexResult<Option<ExtensionTypeRef>> {
        if let Some(ext_type) = V::try_decode(ext_id, metadata)? {
            Ok(Some(ext_type.into_extension_type_ref()))
        } else {
            Ok(None)
        }
    }
}

/// A cheaply cloneable type-erased handle for extension type encodings.
pub type ExtensionTypeEncodingRef = ArcRef<dyn ExtensionTypeEncoding>;

impl dyn ExtensionTypeEncoding + '_ {
    /// Predicate to check if the inner encoding type is the one specified by
    /// the provided VTable.
    pub fn is<V: ExtensionVTable>(&self) -> bool {
        self.as_opt::<V>().is_some()
    }

    /// Force downcast, panicking on failure.
    ///
    /// See also: [`Self::as_opt`]
    pub fn as_<V: ExtensionVTable>(&self) -> &V::ExtEncoding {
        self.as_opt::<V>()
            .vortex_expect("ExtensionTypeEncoding not of expected type")
    }

    /// Try and downcast to a specific encoding variant.
    pub fn as_opt<V: ExtensionVTable>(&self) -> Option<&V::ExtEncoding> {
        self.as_any()
            .downcast_ref::<ExtensionTypeEncodingAdapter<V>>()
            .map(|adapter| &adapter.0)
    }
}
