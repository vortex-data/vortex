use std::fmt::{Debug, Formatter};

use flexbuffers::FlexbufferSerializer;
use vortex_buffer::ByteBuffer;
use vortex_error::{vortex_bail, vortex_err, VortexError, VortexExpect, VortexResult};

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

/// A utility wrapper for automating the serialization of metadata using [rkyv](https://docs.rs/rkyv/latest/rkyv/).
pub struct RkyvMetadata<M>(pub M);

impl<M> SerializeMetadata for RkyvMetadata<M>
where
    M: for<'a> rkyv::Serialize<
        rkyv::api::high::HighSerializer<
            rkyv::util::AlignedVec,
            rkyv::ser::allocator::ArenaHandle<'a>,
            VortexError,
        >,
    >,
{
    fn serialize(&self) -> Option<Vec<u8>> {
        let buf = rkyv::to_bytes::<VortexError>(&self.0)
            .vortex_expect("Failed to serialize metadata using rkyv");
        if buf.is_empty() {
            None
        } else {
            Some(buf.to_vec())
        }
    }
}

impl<M: Debug> Debug for RkyvMetadata<M> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

// TODO(ngates): this is slightly naive and more expensive than necessary.
//  Many cases could use rkyv access instead of deserialize, which allows partial zero-copy
//  access to the metadata. That said... our intention is to move towards u64 metadata, in which
//  case the cost is negligible.
impl<M> DeserializeMetadata for RkyvMetadata<M>
where
    M: Debug,
    M: rkyv::Archive,
    M::Archived:
        for<'a> rkyv::bytecheck::CheckBytes<rkyv::api::high::HighValidator<'a, VortexError>>,
    // Safe deserialization requires a pool
    M::Archived: rkyv::Deserialize<M, rkyv::rancor::Strategy<rkyv::de::Pool, VortexError>>,
    // Unsafe deserialization doesn't require a pool.
    M::Archived: rkyv::Deserialize<M, rkyv::rancor::Strategy<(), VortexError>>,
{
    type Output = M;

    fn deserialize(metadata: Option<&[u8]>) -> VortexResult<Self::Output> {
        rkyv::from_bytes::<M, VortexError>(
            metadata.ok_or_else(|| vortex_err!("Missing expected metadata"))?,
        )
    }

    unsafe fn deserialize_unchecked(metadata: Option<&[u8]>) -> Self::Output {
        unsafe {
            rkyv::api::low::from_bytes_unchecked(
                metadata.vortex_expect("Missing expected metadata"),
            )
            .vortex_expect("Failed to deserialize metadata")
        }
    }

    #[allow(clippy::use_debug)]
    fn format(metadata: Option<&[u8]>, f: &mut Formatter<'_>) -> std::fmt::Result {
        match Self::deserialize(metadata) {
            Ok(m) => write!(f, "{:?}", m),
            Err(_) => write!(f, "Failed to deserialize metadata"),
        }
    }
}

/// A utility wrapper for automating the serialization of metadata using [serde](docs.rs/serde) into [flexbuffers](https://docs.rs/flexbuffers/latest/flexbuffers/).
pub struct SerdeMetadata<M>(pub M);

impl<M> SerializeMetadata for SerdeMetadata<M>
where
    M: serde::Serialize,
{
    fn serialize(&self) -> Option<Vec<u8>> {
        let mut ser = FlexbufferSerializer::new();
        serde::Serialize::serialize(&self.0, &mut ser)
            .vortex_expect("Failed to serialize metadata using serde");
        Some(ser.take_buffer())
    }
}

impl<M> DeserializeMetadata for SerdeMetadata<M>
where
    M: Debug,
    M: for<'m> serde::Deserialize<'m>,
{
    type Output = M;

    fn deserialize(metadata: Option<&[u8]>) -> VortexResult<Self::Output> {
        let bytes =
            metadata.ok_or_else(|| vortex_err!("Serde metadata requires metadata bytes"))?;
        Ok(M::deserialize(flexbuffers::Reader::get_root(bytes)?)?)
    }

    #[allow(clippy::use_debug)]
    fn format(metadata: Option<&[u8]>, f: &mut Formatter<'_>) -> std::fmt::Result {
        match Self::deserialize(metadata) {
            Ok(m) => write!(f, "{:?}", m),
            Err(_) => write!(f, "Failed to deserialize metadata"),
        }
    }
}

impl<M: Debug> Debug for SerdeMetadata<M> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
