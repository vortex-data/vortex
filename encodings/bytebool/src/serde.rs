use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::{SerdeVTable, ValidityHelper, VisitorVTable};
use vortex_array::{ArrayBufferVisitor, ArrayChildVisitor, DeserializeMetadata, EmptyMetadata};
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::{ByteBoolArray, ByteBoolEncoding, ByteBoolVTable};

impl SerdeVTable<ByteBoolVTable> for ByteBoolVTable {
    type Metadata = EmptyMetadata;

    fn metadata(_array: &ByteBoolArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(EmptyMetadata))
    }

    fn build(
        _encoding: &ByteBoolEncoding,
        dtype: &DType,
        len: usize,
        _metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ByteBoolArray> {
        let validity = if children.is_empty() {
            Validity::from(dtype.nullability())
        } else if children.len() == 1 {
            let validity = children.get(0, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 0 or 1 child, got {}", children.len());
        };

        if buffers.len() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", buffers.len());
        }
        let buffer = buffers[0].clone();

        Ok(ByteBoolArray::new(buffer, validity))
    }
}

impl VisitorVTable<ByteBoolVTable> for ByteBoolVTable {
    fn visit_buffers(array: &ByteBoolArray, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(array.buffer());
    }

    fn visit_children(array: &ByteBoolArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(array.validity(), array.len());
    }
}
