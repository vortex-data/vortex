use std::fmt::{Display, Formatter};
use std::sync::Arc;

use crate::{DType, Nullability};

#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct ExtID(Arc<str>);

impl ExtID {
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

#[derive(Debug, Clone, PartialOrd, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExtMetadata(Arc<[u8]>);

impl ExtMetadata {
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

#[derive(Debug, Clone, PartialOrd, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExtDType {
    id: ExtID,
    scalars_dtype: Arc<DType>,
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
    pub fn new(id: ExtID, scalars_dtype: Arc<DType>, metadata: Option<ExtMetadata>) -> Self {
        assert!(
            !matches!(scalars_dtype.as_ref(), &DType::Extension(_)),
            "ExtDType cannot have Extension scalars_dtype"
        );

        Self {
            id,
            scalars_dtype,
            metadata,
        }
    }

    #[inline]
    pub fn id(&self) -> &ExtID {
        &self.id
    }

    #[inline]
    pub fn scalars_dtype(&self) -> &DType {
        self.scalars_dtype.as_ref()
    }

    pub fn with_scalars_nullability(&self, nullability: Nullability) -> Self {
        Self::new(
            self.id.clone(),
            Arc::new(self.scalars_dtype.with_nullability(nullability)),
            self.metadata.clone(),
        )
    }

    #[inline]
    pub fn metadata(&self) -> Option<&ExtMetadata> {
        self.metadata.as_ref()
    }
}
