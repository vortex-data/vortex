use std::fmt::{Debug, Display, Formatter};

use flexbuffers::FlexbufferSerializer;
use vortex_buffer::ByteBuffer;
use vortex_error::{vortex_bail, vortex_err, VortexError, VortexExpect, VortexResult};

pub trait ArrayMetadata: SerializeMetadata + for<'m> DeserializeMetadata<'m> + Display {}

pub trait SerializeMetadata {
    fn serialize(&self) -> VortexResult<Option<ByteBuffer>>;
}

impl SerializeMetadata for () {
    fn serialize(&self) -> VortexResult<Option<ByteBuffer>> {
        Ok(None)
    }
}

pub trait DeserializeMetadata<'m>
where
    Self: Sized,
{
    fn deserialize(metadata: Option<&'m [u8]>) -> VortexResult<Self>;

    /// Deserialize metadata without validation.
    ///
    /// ## Safety
    ///
    /// Those who use this API must be sure to have invoked deserialize at least once before
    /// calling this method.
    unsafe fn deserialize_unchecked(metadata: Option<&'m [u8]>) -> Self {
        Self::deserialize(metadata)
            .vortex_expect("Metadata should have been validated before calling this method")
    }
}

pub struct EmptyMetadata;
impl ArrayMetadata for EmptyMetadata {}

impl SerializeMetadata for EmptyMetadata {
    fn serialize(&self) -> VortexResult<Option<ByteBuffer>> {
        Ok(None)
    }
}

impl DeserializeMetadata<'_> for EmptyMetadata {
    fn deserialize(metadata: Option<&[u8]>) -> VortexResult<Self> {
        if metadata.is_some() {
            vortex_bail!("EmptyMetadata should not have metadata bytes")
        }
        Ok(EmptyMetadata)
    }
}

impl Display for EmptyMetadata {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("EmptyMetadata")
    }
}

/// A utility wrapper for automating the serialization of metadata using rkyv.
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
    fn serialize(&self) -> VortexResult<Option<ByteBuffer>> {
        let buf = rkyv::to_bytes::<VortexError>(&self.0)?;
        if buf.is_empty() {
            Ok(None)
        } else {
            Ok(Some(ByteBuffer::from(buf)))
        }
    }
}

// TODO(ngates): this is slightly naive and more expensive than necessary.
//  Many cases could use rkyv access instead of deserialize, which allows partial zero-copy
//  access to the metadata. That said... our intention is to move towards u64 metadata, in which
//  case the cost is negligible.
impl<'m, M> DeserializeMetadata<'m> for RkyvMetadata<M>
where
    M: rkyv::Archive,
    M::Archived: for<'a> rkyv::bytecheck::CheckBytes<rkyv::api::high::HighValidator<'a, VortexError>>
        + rkyv::Deserialize<M, rkyv::rancor::Strategy<rkyv::de::Pool, VortexError>>,
{
    fn deserialize(metadata: Option<&'m [u8]>) -> VortexResult<Self> {
        rkyv::from_bytes::<M, VortexError>(
            metadata.ok_or_else(|| vortex_err!("Missing expected metadata"))?,
        )
        .map(RkyvMetadata)
    }
}

#[allow(clippy::use_debug)]
impl<M> Display for RkyvMetadata<M>
where
    M: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

pub struct SerdeMetadata<M>(pub M);

impl<M> SerializeMetadata for SerdeMetadata<M>
where
    M: serde::Serialize,
{
    fn serialize(&self) -> VortexResult<Option<ByteBuffer>> {
        let mut ser = FlexbufferSerializer::new();
        serde::Serialize::serialize(&self.0, &mut ser)?;
        Ok(Some(ser.take_buffer().into()))
    }
}

impl<'m, M> DeserializeMetadata<'m> for SerdeMetadata<M>
where
    M: serde::Deserialize<'m>,
{
    fn deserialize(metadata: Option<&'m [u8]>) -> VortexResult<Self> {
        let bytes =
            metadata.ok_or_else(|| vortex_err!("Serde metadata requires metadata bytes"))?;
        Ok(SerdeMetadata(M::deserialize(
            flexbuffers::Reader::get_root(bytes)?,
        )?))
    }
}

#[allow(clippy::use_debug)]
impl<M> Display for SerdeMetadata<M>
where
    M: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

//
// /// Provide default implementation for metadata serialization based on flexbuffers serde.
// impl<M: Serialize> TrySerializeArrayMetadata for M {
//     fn try_serialize_metadata(&self) -> VortexResult<Arc<[u8]>> {
//         let mut ser = FlexbufferSerializer::new();
//         self.serialize(&mut ser)?;
//         Ok(ser.take_buffer().into())
//     }
// }
//
// impl<'de, M: Deserialize<'de>> TryDeserializeArrayMetadata<'de> for M {
//     fn try_deserialize_metadata(metadata: Option<&'de [u8]>) -> VortexResult<Self> {
//         let bytes = metadata.ok_or_else(|| vortex_err!("Array requires metadata bytes"))?;
//         Ok(M::deserialize(Reader::get_root(bytes)?)?)
//     }
// }
