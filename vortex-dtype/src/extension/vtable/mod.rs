mod temporal;

use std::any::Any;
use std::fmt::Debug;
use std::ops::Deref;
use std::sync::Arc;

use arcref::ArcRef;
use vortex_error::VortexResult;

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
    fn try_decode(id: &ExtID, metadata: Option<ExtMetadata>)
    -> VortexResult<Option<Self::ExtType>>;
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
pub trait ExtensionTypeEncoding {
    /// Entrypoint for downcasting to a concrete subtype.
    fn as_any(&self) -> &dyn Any;

    /// ID for the encoding type.
    fn id(&self) -> ExtID {
        ExtID::new(std::any::type_name::<Self>().into())
    }

    /// See if this deserializes into one of the builtin extension types that are supported
    /// by this registry, propagating any deserialization errors.
    ///
    /// An inner `None` is returned if this encoding does not support serde for this extension type.
    fn try_decode(
        &self,
        ext_id: &ExtID,
        metadata: Option<ExtMetadata>,
    ) -> VortexResult<Option<ExtensionTypeRef>>;
}

/// Implement an extension type encoding using a VTable.
pub struct ExtensionTypeEncodingAdapter<V: ExtensionVTable>(V::ExtEncoding);

impl<V: ExtensionVTable> ExtensionTypeEncoding for ExtensionTypeEncodingAdapter<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn try_decode(
        &self,
        ext_id: &ExtID,
        metadata: Option<ExtMetadata>,
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
