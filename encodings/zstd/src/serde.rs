use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::{EncodeVTable, SerdeVTable, VisitorVTable};
use vortex_array::{ArrayBufferVisitor, ArrayChildVisitor, DeserializeMetadata, ProstMetadata};
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::{ZstdArray, ZstdEncoding, ZstdVTable};

/// Metadata for zstd arrays - stores the uncompressed length and original dtype
#[derive(Clone, prost::Message)]
pub struct ZstdMetadata {
    #[prost(uint32, tag = "1")]
    pub uncompressed_len: u32,
}

impl SerdeVTable<ZstdVTable> for ZstdVTable {
    type Metadata = ProstMetadata<ZstdMetadata>;

    fn metadata(array: &ZstdArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(ProstMetadata(ZstdMetadata {
            uncompressed_len: u32::try_from(array.uncompressed_len)?,
        })))
    }

    fn build(
        _encoding: &ZstdEncoding,
        dtype: &DType,
        len: usize,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ZstdArray> {
        let validity = if children.is_empty() {
            Validity::from(dtype.nullability())
        } else if children.len() == 1 {
            let validity = children.get(0, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("ZstdArray expected 0 or 1 child, got {}", children.len());
        };

        if buffers.len() != 1 {
            vortex_bail!(
                "ZstdArray should have exactly 1 buffer, got {}",
                buffers.len()
            );
        }

        let compressed_data = buffers[0].clone();

        Ok(ZstdArray::new(
            compressed_data,
            dtype.clone(),
            metadata.uncompressed_len as usize,
            validity,
        ))
    }
}

impl EncodeVTable<ZstdVTable> for ZstdVTable {
    fn encode(
        _encoding: &<ZstdVTable as vortex_array::vtable::VTable>::Encoding,
        canonical: &vortex_array::Canonical,
        _like: Option<&ZstdArray>,
    ) -> VortexResult<Option<ZstdArray>> {
        let parray = canonical.clone().into_primitive()?;

        Ok(Some(ZstdArray::from_primitive(&parray, 3)?))
    }
}

impl VisitorVTable<ZstdVTable> for ZstdVTable {
    fn visit_buffers(array: &ZstdArray, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(array.compressed_data());
    }

    fn visit_children(array: &ZstdArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(&array.validity, array.len());
    }
}
