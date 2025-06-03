use vortex_array::serde::ArrayChildren;
use vortex_array::vtable::{SerdeVTable, VisitorVTable};
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
            uncompressed_len: array.uncompressed_len as u32,
        })))
    }

    fn build(
        _encoding: &ZstdEncoding,
        dtype: &DType,
        _len: usize,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ZstdArray> {
        if !children.is_empty() {
            vortex_bail!("ZstdArray should have no children, got {}", children.len());
        }

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
        ))
    }
}

impl VisitorVTable<ZstdVTable> for ZstdVTable {
    fn visit_buffers(array: &ZstdArray, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(array.compressed_data());
    }

    fn visit_children(_array: &ZstdArray, _visitor: &mut dyn ArrayChildVisitor) {
        // ZstdArray has no children
    }
}
