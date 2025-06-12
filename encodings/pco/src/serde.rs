use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::{EncodeVTable, SerdeVTable, VisitorVTable};
use vortex_array::{ArrayBufferVisitor, ArrayChildVisitor, ProstMetadata};
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::{PcoArray, PcoEncoding, PcoVTable};

#[derive(Clone, prost::Message)]
pub struct PcoBufferMetadata {
    #[prost(bool, tag = "1")]
    pub is_chunk_meta: bool,
    #[prost(uint64, tag = "2")]
    pub n: u64, // chunk_n for chunks an page_n for pages
}

#[derive(Clone, prost::Message)]
pub struct PcoMetadata {
    #[prost(message, repeated, tag = "1")]
    pub buffers: Vec<PcoBufferMetadata>,
}

impl SerdeVTable<PcoVTable> for PcoVTable {
    type Metadata = ProstMetadata<PcoMetadata>;

    fn metadata(array: &PcoArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(ProstMetadata(array.metadata.clone())))
    }

    fn build(
        _encoding: &PcoEncoding,
        dtype: &DType,
        len: usize,
        metadata: &PcoMetadata,
        buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<PcoArray> {
        let validity = if children.is_empty() {
            Validity::from(dtype.nullability())
        } else if children.len() == 1 {
            let validity = children.get(0, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("PcoArray expected 0 or 1 child, got {}", children.len());
        };

        Ok(PcoArray::new(
            buffers,
            dtype.clone(),
            metadata.clone(),
            len,
            validity,
        ))
    }
}

impl EncodeVTable<PcoVTable> for PcoVTable {
    fn encode(
        _encoding: &<PcoVTable as vortex_array::vtable::VTable>::Encoding,
        canonical: &vortex_array::Canonical,
        _like: Option<&PcoArray>,
    ) -> VortexResult<Option<PcoArray>> {
        let parray = canonical.clone().into_primitive()?;

        Ok(Some(PcoArray::from_primitive(&parray, 3, 0)?))
    }
}

impl VisitorVTable<PcoVTable> for PcoVTable {
    fn visit_buffers(array: &PcoArray, visitor: &mut dyn ArrayBufferVisitor) {
        for buffer in &array.buffers {
            visitor.visit_buffer(&buffer.inner);
        }
    }

    fn visit_children(array: &PcoArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(&array.validity, array.len());
    }
}
