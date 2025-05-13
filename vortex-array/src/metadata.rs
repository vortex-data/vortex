use std::fmt::{Debug, Formatter};
use std::ops::Deref;

use vortex_error::{VortexResult, vortex_bail};

/// Trait for serializing Vortex metadata to a vector of unaligned bytes.
pub trait SerializeMetadata {
    fn serialize(self) -> Vec<u8>;
}

/// Trait for deserializing Vortex metadata from a vector of unaligned bytes.
pub trait DeserializeMetadata
where
    Self: Sized,
{
    /// The fully deserialized type of the metadata.
    type Output;

    /// Deserialize metadata from a vector of unaligned bytes.
    fn deserialize(metadata: &[u8]) -> VortexResult<Self::Output>;
}

/// Empty array metadata
#[derive(Debug)]
pub struct EmptyMetadata;

impl SerializeMetadata for EmptyMetadata {
    fn serialize(self) -> Vec<u8> {
        vec![]
    }
}

impl DeserializeMetadata for EmptyMetadata {
    type Output = EmptyMetadata;

    fn deserialize(metadata: &[u8]) -> VortexResult<Self::Output> {
        if !metadata.is_empty() {
            vortex_bail!("EmptyMetadata should not have metadata bytes")
        }
        Ok(EmptyMetadata)
    }
}

/// A utility wrapper for raw metadata serialization. This delegates the serialiation step
/// to the arrays' vtable.
pub struct RawMetadata(pub Vec<u8>);

impl SerializeMetadata for RawMetadata {
    fn serialize(self) -> Vec<u8> {
        self.0
    }
}

impl DeserializeMetadata for RawMetadata {
    type Output = Vec<u8>;

    fn deserialize(metadata: &[u8]) -> VortexResult<Self::Output> {
        Ok(metadata.to_vec())
    }
}

impl Debug for RawMetadata {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "\"{}\"", self.0.escape_ascii())
    }
}

/// A utility wrapper for Prost metadata serialization.
pub struct ProstMetadata<M>(pub M);

impl<M> Deref for ProstMetadata<M> {
    type Target = M;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<M: Debug> Debug for ProstMetadata<M> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl<M> SerializeMetadata for ProstMetadata<M>
where
    M: prost::Message,
{
    fn serialize(self) -> Vec<u8> {
        self.0.encode_to_vec()
    }
}

impl<M> DeserializeMetadata for ProstMetadata<M>
where
    M: Debug,
    M: prost::Message + Default,
{
    type Output = M;

    fn deserialize(metadata: &[u8]) -> VortexResult<Self::Output> {
        Ok(M::decode(metadata)?)
    }
}
