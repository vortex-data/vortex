use std::fmt::{Display, Formatter};
use std::sync::Arc;

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
#[derive(Debug, Clone, PartialOrd, PartialEq, Eq, Hash)]
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

/// A type descriptor for an extension type
#[derive(Debug, Clone, PartialOrd, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExtDType {
    id: ExtID,
    metadata: Option<ExtMetadata>,
}

impl ExtDType {
    /// Constructs a new `ExtDType` from an `ExtID` and optional `ExtMetadata`
    pub fn new(id: ExtID, metadata: Option<ExtMetadata>) -> Self {
        Self { id, metadata }
    }

    /// Returns the `ExtID` for this extension type
    #[inline]
    pub fn id(&self) -> &ExtID {
        &self.id
    }

    /// Returns the `ExtMetadata` for this extension type, if it exists
    #[inline]
    pub fn metadata(&self) -> Option<&ExtMetadata> {
        self.metadata.as_ref()
    }
}
