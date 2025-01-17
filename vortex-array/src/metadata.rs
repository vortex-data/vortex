use vortex_buffer::ByteBuffer;
use vortex_error::{vortex_err, VortexError, VortexExpect, VortexResult};

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
