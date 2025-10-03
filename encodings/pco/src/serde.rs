// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::{EncodeVTable, SerdeVTable, VisitorVTable};
use vortex_array::{ArrayBufferVisitor, ArrayChildVisitor, ProstMetadata};
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail, vortex_ensure};

use crate::{PcoArray, PcoEncoding, PcoVTable};

#[derive(Clone, prost::Message)]
pub struct PcoPageInfo {
    // Since pco limits to 2^24 values per chunk, u32 is sufficient for the
    // count of values.
    #[prost(uint32, tag = "1")]
    pub n_values: u32,
}

// We're calling this Info instead of Metadata because ChunkMeta refers to a specific
// component of a Pco file.
#[derive(Clone, prost::Message)]
pub struct PcoChunkInfo {
    #[prost(message, repeated, tag = "1")]
    pub pages: Vec<PcoPageInfo>,
}

#[derive(Clone, prost::Message)]
pub struct PcoMetadata {
    // would be nice to reuse one header per vortex file, but it's really only 1 byte, so
    // no issue duplicating it here per PcoArray
    #[prost(bytes, tag = "1")]
    pub header: Vec<u8>,
    #[prost(message, repeated, tag = "2")]
    pub chunks: Vec<PcoChunkInfo>,
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

        vortex_ensure!(buffers.len() >= metadata.chunks.len());
        let chunk_metas = buffers[..metadata.chunks.len()].to_vec();
        let pages = buffers[metadata.chunks.len()..].to_vec();

        let expected_n_pages = metadata
            .chunks
            .iter()
            .map(|info| info.pages.len())
            .sum::<usize>();
        vortex_ensure!(pages.len() == expected_n_pages);

        Ok(PcoArray::new(
            chunk_metas,
            pages,
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
        let parray = canonical.clone().into_primitive();

        Ok(Some(PcoArray::from_primitive(&parray, 3, 0)?))
    }
}

impl VisitorVTable<PcoVTable> for PcoVTable {
    fn visit_buffers(array: &PcoArray, visitor: &mut dyn ArrayBufferVisitor) {
        for buffer in &array.chunk_metas {
            visitor.visit_buffer(buffer);
        }
        for buffer in &array.pages {
            visitor.visit_buffer(buffer);
        }
    }

    fn visit_children(array: &PcoArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(&array.unsliced_validity, array.unsliced_n_rows());
    }
}
