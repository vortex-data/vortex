use std::any::Any;
use std::fmt::{Debug, Display};
use std::io::Cursor;
use std::sync::Arc;

use vortex_avro::{FromAvro, ToAvro};
use vortex_error::{vortex_err, VortexResult};

use crate::encoding::Encoding;

/// Dynamic trait used to represent opaque owned Array metadata
///
/// Note that this allows us to restrict the ('static + Send + Sync) requirement to just the
/// metadata trait, and not the entire array trait. We require 'static so that we can downcast
/// use the Any trait.
pub trait ArrayMetadata:
    'static + Send + Sync + Debug + TrySerializeArrayMetadata + Display
{
    fn as_any(&self) -> &dyn Any;
    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync>;
}

pub trait GetArrayMetadata {
    fn metadata(&self) -> Arc<dyn ArrayMetadata>;
}

pub trait TrySerializeArrayMetadata {
    fn try_serialize_metadata(&self) -> VortexResult<Arc<[u8]>>;
}

pub trait TryDeserializeArrayMetadata<'m>: Sized {
    fn try_deserialize_metadata(metadata: Option<&'m [u8]>) -> VortexResult<Self>;
}

/// Provide default implementation for metadata serialization based on flexbuffers serde.
impl<M: ToAvro + Clone> TrySerializeArrayMetadata for M {
    fn try_serialize_metadata(&self) -> VortexResult<Arc<[u8]>> {
        vortex_avro::to_avro_binary(self.clone()).map(Arc::from)
    }
}

impl<'de, M: FromAvro> TryDeserializeArrayMetadata<'de> for M {
    fn try_deserialize_metadata(metadata: Option<&'de [u8]>) -> VortexResult<Self> {
        let bytes: &[u8] = metadata.ok_or_else(|| vortex_err!("Array requires metadata bytes"))?;
        let mut reader = Cursor::new(bytes);
        vortex_avro::from_avro_binary(&M::read_schema(), &mut reader)
    }
}

pub trait MetadataVTable {
    fn load_metadata(&self, metadata: Option<&[u8]>) -> VortexResult<Arc<dyn ArrayMetadata>>;
}

impl<E: Encoding> MetadataVTable for E
where
    E::Metadata: for<'m> TryDeserializeArrayMetadata<'m>,
{
    fn load_metadata(&self, metadata: Option<&[u8]>) -> VortexResult<Arc<dyn ArrayMetadata>> {
        E::Metadata::try_deserialize_metadata(metadata)
            .map(|m| Arc::new(m) as Arc<dyn ArrayMetadata>)
    }
}
