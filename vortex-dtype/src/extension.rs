use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::sync::{Arc, LazyLock, RwLock};

use vortex_error::{VortexExpect, VortexResult};

use crate::{DType, Nullability};

/// A unique identifier for an extension type
#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct ExtID(Arc<str>);

impl ExtID {
    /// Constructs a new `ExtID` from a string
    pub fn new(value: Arc<str>) -> Self {
        Self(value)
    }
}

impl Display for ExtID {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl AsRef<str> for ExtID {
    fn as_ref(&self) -> &str {
        self.0.as_ref()
    }
}

impl From<&str> for ExtID {
    fn from(value: &str) -> Self {
        Self(value.into())
    }
}

/// Opaque metadata for an extension type
#[derive(Debug, Default, Clone, PartialOrd, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExtMetadata(Arc<[u8]>);

impl ExtMetadata {
    /// Constructs a new `ExtMetadata` from a byte slice
    pub fn new(value: Arc<[u8]>) -> Self {
        Self(value)
    }
}

impl AsRef<[u8]> for ExtMetadata {
    fn as_ref(&self) -> &[u8] {
        self.0.as_ref()
    }
}

impl From<&[u8]> for ExtMetadata {
    fn from(value: &[u8]) -> Self {
        Self(value.into())
    }
}

/// A type descriptor for an extension type.
///
/// Stores the type ID, the logical type of the stored values, and an optional
/// piece of metadata.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExtDType {
    id: ExtID,
    storage_dtype: Arc<DType>,
    metadata: Option<ExtMetadata>,
}

impl ExtDType {
    /// Creates a new `ExtDType`.
    ///
    /// Extension data types in Vortex allows library users to express additional semantic meaning
    /// on top of a set of scalar values. Metadata can optionally be provided for the extension type
    /// to allow for parameterized types.
    ///
    /// A simple example would be if one wanted to create a `vortex.temperature` extension type. The
    /// canonical encoding for such values would be `f64`, and the metadata can contain an optional
    /// temperature unit, allowing downstream users to be sure they properly account for Celsius
    /// and Fahrenheit conversions.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use vortex_dtype::{DType, ExtDType, ExtID, ExtMetadata, Nullability, PType};
    ///
    /// #[repr(u8)]
    /// enum TemperatureUnit {
    ///     C = 0u8,
    ///     F = 1u8,
    /// }
    ///
    /// // Make a new extension type that encodes the unit for a set of nullable `f64`.
    /// pub fn create_temperature_type(unit: TemperatureUnit) -> ExtDType {
    ///     ExtDType::new(
    ///         ExtID::new("vortex.temperature".into()),
    ///         Arc::new(DType::Primitive(PType::F64, Nullability::Nullable)),
    ///         Some(ExtMetadata::new([unit as u8].into()))
    ///     )
    /// }
    /// ```
    pub fn new(id: ExtID, storage_dtype: Arc<DType>, metadata: Option<ExtMetadata>) -> Self {
        assert!(
            !matches!(storage_dtype.as_ref(), &DType::Extension(_)),
            "ExtDType cannot have Extension storage_dtype"
        );

        Self {
            id,
            storage_dtype,
            metadata,
        }
    }

    /// Returns the `ExtID` for this extension type
    #[inline]
    pub fn id(&self) -> &ExtID {
        &self.id
    }

    /// Returns the `ExtMetadata` for this extension type, if it exists
    #[inline]
    pub fn storage_dtype(&self) -> &DType {
        self.storage_dtype.as_ref()
    }

    /// Returns a new `ExtDType` with the given nullability
    pub fn with_nullability(&self, nullability: Nullability) -> Self {
        Self::new(
            self.id.clone(),
            Arc::new(self.storage_dtype.with_nullability(nullability)),
            self.metadata.clone(),
        )
    }

    /// Returns the `ExtMetadata` for this extension type, if it exists
    #[inline]
    pub fn metadata(&self) -> Option<&ExtMetadata> {
        self.metadata.as_ref()
    }

    /// Check if `self` and `other` are equal, ignoring the storage nullability
    pub fn eq_ignore_nullability(&self, other: &Self) -> bool {
        self.id() == other.id()
            && self.metadata() == other.metadata()
            && self
                .storage_dtype()
                .eq_ignore_nullability(other.storage_dtype())
    }

    pub fn try_extension_type<T: ExtensionType>(&self) -> Option<T> {
        if !ExtensionRegistry::shared().is_valid(self) {
            None
        } else {
            ExtensionRegistry::shared().is_valid()
        }
    }
}

/// A trait covering pluggable extension types.
///
/// Any user of this crate can create their own extension types and plug them into Vortex by defining
/// an implementation of this trait for their type. The trait defines an interface for types to
/// be in serialized/deserialized form, as well as some of the other metadata types here.
pub trait ExtensionType {
    /// Type of the metadata attached to this type.
    ///
    /// If the type has not metadata `()` can be used.
    type Metadata;

    /// Globally unique type identifier for the extension.
    fn type_id() -> ExtID;

    /// Get a reference to the metadata for an instance of the extension type.
    fn metadata(&self) -> &Self::Metadata;

    /// Serialize the owned metadata into an `ExtMetadata` serialized format.
    fn serialize(&self) -> Option<ExtMetadata>;

    /// Deserialize a piece of owned metadata from the serialized `ExtMetadata`, propagating
    /// any errors in the deserialization process.
    fn try_deserialize(serialized: &ExtMetadata) -> VortexResult<Self::Metadata>
    where
        Self: Sized;

    /// Create a new extension type instance from the storage type and metadata.
    fn try_new(storage_type: DType, metadata: Self::Metadata) -> VortexResult<Self>
    where
        Self: Sized;

    #[cfg(feature = "arrow")]
    /// Convert to an Arrow [`Field`] containing the data type and any Arrow extension metadata
    /// that should be attached for Arrow clients.
    ///
    /// By default nothing is returned.
    fn to_field(&self) -> Option<arrow_schema::Field> {
        None
    }
}

#[cfg(feature = "arrow")]
mod arrow_impl {
    use arrow_schema::extension::ExtensionType as ArrowExtensionType;
    use arrow_schema::{DataType, Field};
    use vortex_error::{VortexResult, vortex_err};

    use crate::{DType, ExtID, ExtMetadata, ExtensionType};

    pub struct GenericArrowExtensionType<T: ArrowExtensionType> {
        data_type: DataType,
        extension_type: T,
    }

    // If it works, we can capture one from somewhere and deploy it here instead.
    // If we're getting some sort of type here we can capture the extension type so
    // that we get access to the type values here instead...I think?

    // Arrow uses std HashMap which we otherwise disallow.
    #[allow(clippy::disallowed_types)]
    impl<T: ArrowExtensionType> ExtensionType for T {
        type Metadata = <T as ArrowExtensionType>::Metadata;

        fn type_id() -> ExtID {
            ExtID::from(T::NAME)
        }

        fn metadata(&self) -> &Self::Metadata {
            <T as ArrowExtensionType>::metadata(self)
        }

        fn serialize(&self) -> Option<ExtMetadata> {
            Some(ExtMetadata::from(T::serialize_metadata(self)?.as_bytes()))
        }

        fn try_deserialize(metadata: &ExtMetadata) -> VortexResult<Self::Metadata> {
            let bytes = metadata.as_ref();
            let json = std::str::from_utf8(bytes)
                .map_err(|e| vortex_err!("failed to parse metadata as UTF-8 string: {}", e))?;
            // Figure out how to deserialize the metadata as-is here.
            Ok(T::deserialize_metadata(Some(json))?)
        }

        fn try_new(storage_type: DType, metadata: Self::Metadata) -> VortexResult<Self>
        where
            Self: Sized,
        {
            Ok(<T as ArrowExtensionType>::try_new(
                &storage_type.to_arrow()?,
                metadata,
            )?)
        }

        /// Convert self into a field type.
        /// If we have an Arrow extension type wrapping another type, then that works as expected.
        fn to_field(&self) -> Option<Field> {
            todo!()
        }
    }
}

/// Validation function, constructed from a T to validate that the metadata and storage
/// type conform to the requirements of the type.
type ValidateFn = Box<dyn Fn(&ExtDType) -> VortexResult<()> + Send + Sync + 'static>;

#[cfg(feature = "arrow")]
/// Convert the a storage type + metadata into some inner T and then convert it to a field.
type ToFieldFn =
    Box<dyn Fn(&DType, &ExtMetadata) -> Option<arrow_schema::Field> + Send + Sync + 'static>;

// Entry in the extensions table
struct Entry {
    validate_fn: ValidateFn,
    #[cfg(feature = "arrow")]
    to_field_fn: ToFieldFn,
}

static REGISTRY: LazyLock<ExtensionRegistry> = LazyLock::new(|| ExtensionRegistry {
    extensions: Arc::new(RwLock::new(HashMap::new())),
});

pub struct ExtensionRegistry {
    extensions: Arc<RwLock<HashMap<ExtID, Entry>>>,
}

impl ExtensionRegistry {
    /// Get access to the shared registry.
    pub fn shared() -> &'static Self {
        &REGISTRY
    }

    /// Register the extension type `T` so that it can be recognized by several intrinsic Vortex
    /// functions that operate over extensions.
    pub fn register<T: ExtensionType>(&self) {
        let validate_fn = Box::new(|ext_dtype: &ExtDType| {
            let m = T::try_deserialize(&ext_dtype.metadata().cloned().unwrap_or_default())?;
            let _ = T::try_new(ext_dtype.storage_dtype().clone(), m)?;
            Ok(())
        });

        #[cfg(feature = "arrow")]
        let to_field_fn = Box::new(|storage_type: &DType, metadata: &ExtMetadata| {
            let m = T::try_deserialize(&metadata)?;
            let ext = T::try_new(storage_type.clone(), m)
                .vortex_expect("validation should have succeeded previously");
            ext.to_field()
        });

        self.extensions.write().unwrap().insert(
            T::type_id(),
            Entry {
                validate_fn,
                #[cfg(feature = "arrow")]
                to_field_fn,
            },
        );
    }

    /// Validate that the metadata is valid according to our validation logic.
    pub fn is_valid(&self, extension_type: &ExtDType) -> bool {
        let map_guard = self.extensions.read().vortex_expect("poisoned");
        if let Some(entry) = map_guard.get(extension_type.id()) {
            (entry.validate_fn)(
                extension_type.storage_dtype(),
                &extension_type.metadata().cloned().unwrap_or_default(),
            )
            .is_ok()
        } else {
            false
        }
    }

    #[cfg(feature = "arrow")]
    pub fn try_to_arrow(&self, ext_dtype: &ExtDType) -> Option<arrow_schema::Field> {
        if let Some(entry) = self
            .extensions
            .read()
            .vortex_expect("poisoned")
            .get(ext_dtype.id())
        {
            (entry.to_field_fn)()
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use super::{ExtDType, ExtID};
    use crate::{DType, Nullability, PType};

    #[test]
    fn different_ids_are_not_equal() {
        let storage_dtype = Arc::from(DType::Bool(Nullability::NonNullable));
        let one = ExtDType::new(ExtID::new(Arc::from("one")), storage_dtype.clone(), None);
        let two = ExtDType::new(ExtID::new(Arc::from("two")), storage_dtype, None);

        assert_ne!(one, two);
    }

    #[test]
    fn same_id_different_storage_types_are_not_equal() {
        let one = ExtDType::new(
            ExtID::new(Arc::from("one")),
            Arc::from(DType::Bool(Nullability::NonNullable)),
            None,
        );
        let two = ExtDType::new(
            ExtID::new(Arc::from("one")),
            Arc::from(DType::Primitive(PType::U8, Nullability::NonNullable)),
            None,
        );

        assert_ne!(one, two);
    }

    #[test]
    fn same_id_different_nullability_are_not_equal() {
        let nullable_u8 = Arc::from(DType::Primitive(PType::U8, Nullability::NonNullable));
        let one = ExtDType::new(ExtID::new(Arc::from("one")), nullable_u8.clone(), None);
        let two = ExtDType::new(
            ExtID::new(Arc::from("one")),
            Arc::from(nullable_u8.as_nullable()),
            None,
        );

        assert_ne!(one, two);
    }
}
