use vortex_error::VortexResult;

pub trait DeserializeMetadata {
    type Metadata<'m>;

    fn deserialize(metadata: Option<&[u8]>) -> VortexResult<Self::Metadata<'_>>;
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
