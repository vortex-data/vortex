// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::{EncodeVTable, SerdeVTable, VisitorVTable};
use vortex_array::{ArrayBufferVisitor, ArrayChildVisitor, ProstMetadata};
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::{ZstdArray, ZstdEncoding, ZstdVTable};

#[derive(Clone, prost::Message)]
pub struct ZstdFrameMetadata {
    #[prost(uint64, tag = "1")]
    pub uncompressed_size: u64,
    #[prost(uint64, tag = "2")]
    pub n_values: u64,
}

#[derive(Clone, prost::Message)]
pub struct ZstdMetadata {
    // optional, will be 0 if there's no dictionary
    #[prost(uint32, tag = "1")]
    pub dictionary_size: u32,
    #[prost(message, repeated, tag = "2")]
    pub frames: Vec<ZstdFrameMetadata>,
}

impl SerdeVTable<ZstdVTable> for ZstdVTable {
    type Metadata = ProstMetadata<ZstdMetadata>;

    fn metadata(array: &ZstdArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(ProstMetadata(array.metadata.clone())))
    }

    fn build(
        _encoding: &ZstdEncoding,
        dtype: &DType,
        len: usize,
        metadata: &ZstdMetadata,
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

        let (dictionary_buffer, compressed_buffers) = if metadata.dictionary_size == 0 {
            // no dictionary
            (None, buffers.to_vec())
        } else {
            // with dictionary
            (Some(buffers[0].clone()), buffers[1..].to_vec())
        };

        Ok(ZstdArray::new(
            dictionary_buffer,
            compressed_buffers,
            dtype.clone(),
            metadata.clone(),
            len,
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
        ZstdArray::from_canonical(canonical, 3, 0)
    }
}

impl VisitorVTable<ZstdVTable> for ZstdVTable {
    fn visit_buffers(array: &ZstdArray, visitor: &mut dyn ArrayBufferVisitor) {
        if let Some(buffer) = &array.dictionary {
            visitor.visit_buffer(buffer);
        }
        for buffer in &array.frames {
            visitor.visit_buffer(buffer);
        }
    }

    fn visit_children(array: &ZstdArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(&array.unsliced_validity, array.unsliced_n_rows());
    }
}
