// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::{CompressedBoolArray, CompressedBoolEncoding, CompressedBoolVTable};
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::{SerdeVTable, ValidityHelper, VisitorVTable};
use vortex_array::{ArrayBufferVisitor, ArrayChildVisitor, DeserializeMetadata, ProstMetadata};
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

#[derive(Clone, prost::Message)]
pub struct CompressedBoolMetadata {
    #[prost(uint32, tag = "1")]
    pub(crate) bit_offset: u32, // must be <8
}

impl SerdeVTable<CompressedBoolVTable> for CompressedBoolVTable {
    type Metadata = ProstMetadata<CompressedBoolMetadata>;

    fn metadata(array: &CompressedBoolArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(ProstMetadata(CompressedBoolMetadata {
            bit_offset: u32::try_from(array.bit_offset())
                .vortex_expect("bit_offset must fit in u32"),
        })))
    }

    fn build(
        _encoding: &CompressedBoolEncoding,
        dtype: &DType,
        len: usize,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<CompressedBoolArray> {
        if children.is_empty() {
            vortex_bail!("Expected 1 or 2 children, got {}", children.len());
        }
        let byte_length = (metadata.bit_offset as usize + len).div_ceil(8);
        let compressed_buffer = children.get(
            0,
            &DType::Primitive(PType::U8, Nullability::NonNullable),
            byte_length,
        )?;
        let validity = if children.len() == 1 {
            Validity::from(dtype.nullability())
        } else if children.len() == 2 {
            Validity::Array(children.get(0, &Validity::DTYPE, len)?)
        } else {
            vortex_bail!("Expected 1 or 2 children, got {}", children.len());
        };

        if buffers.len() != 0 {
            vortex_bail!("Expected 0 buffers, got {}", buffers.len());
        }

        Ok(CompressedBoolArray::try_new(
            compressed_buffer,
            validity,
            metadata.bit_offset as usize,
            len,
        )?)
    }
}

impl VisitorVTable<CompressedBoolVTable> for CompressedBoolVTable {
    fn visit_buffers(_array: &CompressedBoolArray, _visitor: &mut dyn ArrayBufferVisitor) {}
    fn visit_children(array: &CompressedBoolArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("compressed_buffer", array.compressed_buffer());
        visitor.visit_validity(array.validity(), array.len());
    }
}
