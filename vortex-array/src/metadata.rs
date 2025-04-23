use std::fmt::{Debug, Formatter};

use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};

pub trait SerializeMetadata {
    fn serialize(&self) -> Option<Vec<u8>>;
}

impl SerializeMetadata for () {
    fn serialize(&self) -> Option<Vec<u8>> {
        None
    }
}

pub trait DeserializeMetadata
where
    Self: Sized,
{
    type Output;

    fn deserialize(metadata: Option<&[u8]>) -> VortexResult<Self::Output>;

    /// Deserialize metadata without validation.
    ///
    /// ## Safety
    ///
    /// Those who use this API must be sure to have invoked deserialize at least once before
    /// calling this method.
    unsafe fn deserialize_unchecked(metadata: Option<&[u8]>) -> Self::Output {
        Self::deserialize(metadata)
            .vortex_expect("Metadata should have been validated before calling this method")
    }

    /// Format metadata for display.
    fn format(metadata: Option<&[u8]>, f: &mut Formatter<'_>) -> std::fmt::Result;
}

/// Empty array metadata
#[derive(Debug)]
pub struct EmptyMetadata;

impl SerializeMetadata for EmptyMetadata {
    fn serialize(&self) -> Option<Vec<u8>> {
        None
    }
}

impl DeserializeMetadata for EmptyMetadata {
    type Output = EmptyMetadata;

    fn deserialize(metadata: Option<&[u8]>) -> VortexResult<Self::Output> {
        if metadata.is_some() {
            vortex_bail!("EmptyMetadata should not have metadata bytes")
        }
        Ok(EmptyMetadata)
    }

    fn format(_metadata: Option<&[u8]>, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("EmptyMetadata")
    }
}

/// A utility wrapper for Prost metadata serialization.
pub struct ProstMetadata<M>(pub M);

impl<M: Debug> Debug for ProstMetadata<M> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl<M> SerializeMetadata for ProstMetadata<M>
where
    M: prost::Message,
{
    fn serialize(&self) -> Option<Vec<u8>> {
        Some(self.0.encode_to_vec())
    }
}

impl<M> DeserializeMetadata for ProstMetadata<M>
where
    M: Debug,
    M: prost::Message + Default,
{
    type Output = M;

    fn deserialize(metadata: Option<&[u8]>) -> VortexResult<Self::Output> {
        let bytes =
            metadata.ok_or_else(|| vortex_err!("Prost metadata requires metadata bytes"))?;
        Ok(M::decode(bytes)?)
    }

    #[allow(clippy::use_debug)]
    fn format(metadata: Option<&[u8]>, f: &mut Formatter<'_>) -> std::fmt::Result {
        match Self::deserialize(metadata) {
            Ok(m) => write!(f, "{:?}", m),
            Err(_) => write!(f, "Failed to deserialize metadata"),
        }
    }
}
