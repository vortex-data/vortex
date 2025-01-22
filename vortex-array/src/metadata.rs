use std::fmt::{Debug, Display, Formatter};

use flexbuffers::FlexbufferSerializer;
use vortex_buffer::ByteBuffer;
use vortex_dtype::{ToBytes, TryFromBytes};
use vortex_error::{vortex_bail, vortex_err, VortexError, VortexExpect, VortexResult};

use crate::encoding::Encoding;
use crate::{metadata, ArrayData};

pub type MetadataBytes = [u8; 8];

pub trait ArrayMetadata: SerializeMetadata + DeserializeMetadata + Display {}

pub trait SerializeMetadata {
    fn serialize(&self) -> VortexResult<MetadataBytes>;
}

impl SerializeMetadata for () {
    fn serialize(&self) -> VortexResult<MetadataBytes> {
        Ok([0; 8])
    }
}

pub trait DeserializeMetadata
where
    Self: Sized,
{
    type Output;

    fn deserialize(metadata: MetadataBytes) -> VortexResult<Self::Output>;

    /// Deserialize metadata without validation.
    ///
    /// ## Safety
    ///
    /// Those who use this API must be sure to have invoked deserialize at least once before
    /// calling this method.
    unsafe fn deserialize_unchecked(metadata: MetadataBytes) -> Self::Output {
        Self::deserialize(metadata)
            .vortex_expect("Metadata should have been validated before calling this method")
    }

    /// Format metadata for display.
    fn format(metadata: MetadataBytes, f: &mut Formatter<'_>) -> std::fmt::Result;
}

pub trait MetadataVTable<Array> {
    fn validate_metadata(&self, metadata: MetadataBytes) -> VortexResult<()>;

    fn display_metadata(&self, array: &Array, f: &mut Formatter<'_>) -> std::fmt::Result;
}

impl<E: Encoding> MetadataVTable<ArrayData> for E {
    fn validate_metadata(&self, metadata: MetadataBytes) -> VortexResult<()> {
        E::Metadata::deserialize(metadata).map(|_| ())
    }

    fn display_metadata(&self, array: &ArrayData, f: &mut Formatter<'_>) -> std::fmt::Result {
        <E::Metadata as DeserializeMetadata>::format(array.metadata_bytes(), f)
    }
}

pub struct EmptyMetadata;
impl ArrayMetadata for EmptyMetadata {}

impl SerializeMetadata for EmptyMetadata {
    fn serialize(&self) -> VortexResult<MetadataBytes> {
        Ok([0; 8])
    }
}

impl DeserializeMetadata for EmptyMetadata {
    type Output = EmptyMetadata;

    fn deserialize(_metadata: MetadataBytes) -> VortexResult<Self::Output> {
        Ok(EmptyMetadata)
    }

    fn format(_metadata: MetadataBytes, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("EmptyMetadata")
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
    fn serialize(&self) -> VortexResult<[u8; 8]> {
        let buf = rkyv::to_bytes::<VortexError>(&self.0)?;
        if buf.len() > 8 {
            vortex_bail!("Metadata exceeds 8 bytes")
        }
        let mut metadata: [u8; 8] = [0; 8];
        metadata[..buf.len()].copy_from_slice(buf.as_slice());
        Ok(metadata)
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
    M::Archived: for<'a> rkyv::bytecheck::CheckBytes<rkyv::api::high::HighValidator<'a, VortexError>>
        + rkyv::Deserialize<M, rkyv::rancor::Strategy<rkyv::de::Pool, VortexError>>,
{
    type Output = M;

    fn deserialize(metadata: MetadataBytes) -> VortexResult<Self::Output> {
        rkyv::from_bytes::<M, VortexError>(&metadata[..])
    }

    #[allow(clippy::use_debug)]
    fn format(metadata: MetadataBytes, f: &mut Formatter<'_>) -> std::fmt::Result {
        match Self::deserialize(metadata) {
            Ok(m) => write!(f, "{:?}", m),
            Err(_) => write!(f, "Failed to deserialize metadata"),
        }
    }
}

pub struct SerdeMetadata<M>(pub M);

impl<M> SerializeMetadata for SerdeMetadata<M>
where
    M: serde::Serialize,
{
    fn serialize(&self) -> VortexResult<MetadataBytes> {
        let mut ser = FlexbufferSerializer::new();
        serde::Serialize::serialize(&self.0, &mut ser)?;
        let buf = ser.take_buffer();
        if buf.len() > 8 {
            vortex_bail!("Metadata exceeds 8 bytes")
        }
        Ok(buf.as_slice().try_into()?)
    }
}

impl<M> DeserializeMetadata for SerdeMetadata<M>
where
    M: Debug,
    M: for<'m> serde::Deserialize<'m>,
{
    type Output = M;

    fn deserialize(metadata: MetadataBytes) -> VortexResult<Self::Output> {
        Ok(M::deserialize(flexbuffers::Reader::get_root(
            &metadata[..],
        )?)?)
    }

    #[allow(clippy::use_debug)]
    fn format(metadata: MetadataBytes, f: &mut Formatter<'_>) -> std::fmt::Result {
        match Self::deserialize(metadata) {
            Ok(m) => write!(f, "{:?}", m),
            Err(_) => write!(f, "Failed to deserialize metadata"),
        }
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
