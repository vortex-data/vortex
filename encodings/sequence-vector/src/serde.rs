use vortex_array::serde::ArrayChildren;
use vortex_array::vtable::{EncodeVTable, SerdeVTable};
use vortex_array::{Canonical, DeserializeMetadata, ProstMetadata};
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};
use vortex_proto::scalar::ScalarValue;

use crate::array::{SequenceArray, SequenceEncoding, SequenceVTable};

#[derive(Clone, prost::Message)]
pub struct SequenceMetadata {
    #[prost(message, tag = "1")]
    base: Option<ScalarValue>,
    #[prost(message, tag = "2")]
    multiplier: Option<ScalarValue>,
}

impl EncodeVTable<SequenceVTable> for SequenceVTable {
    fn encode(
        _encoding: &SequenceEncoding,
        _canonical: &Canonical,
        _like: Option<&SequenceArray>,
    ) -> VortexResult<Option<SequenceArray>> {
        // TODO(joe): hook up compressor
        Ok(None)
    }
}

impl SerdeVTable<SequenceVTable> for SequenceVTable {
    type Metadata = ProstMetadata<SequenceMetadata>;

    fn metadata(array: &SequenceArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(ProstMetadata(SequenceMetadata {
            base: Some(array.base().into()),
            multiplier: Some(array.multiplier().into()),
        })))
    }

    fn build(
        _encoding: &SequenceEncoding,
        dtype: &DType,
        len: usize,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _buffers: &[ByteBuffer],
        _children: &dyn ArrayChildren,
    ) -> VortexResult<SequenceArray> {
        let ptype = dtype.as_ptype();
        let base = metadata.base.as_ref().vortex_expect("base required");
        let multiplier = metadata
            .multiplier
            .as_ref()
            .vortex_expect("multiplier required");

        Ok(SequenceArray::unchecked_new(
            base.try_into()?,
            multiplier.try_into()?,
            ptype,
            len,
        ))
    }
}
