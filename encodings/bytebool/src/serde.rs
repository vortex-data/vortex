use vortex_array::serde::ArrayParts;
use vortex_array::validity::Validity;
use vortex_array::vtable::{SerdeVTable, ValidityHelper, VisitorVTable};
use vortex_array::{
    ArrayBufferVisitor, ArrayChildVisitor, ArrayContext, ArrayRef, DeserializeMetadata,
    EmptyMetadata,
};
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::{ByteBoolArray, ByteBoolEncoding, ByteBoolVTable};

impl SerdeVTable<ByteBoolVTable> for ByteBoolVTable {
    type Metadata = EmptyMetadata;

    fn metadata(_array: &ByteBoolArray) -> Option<Self::Metadata> {
        Some(EmptyMetadata)
    }

    fn decode(
        _encoding: &ByteBoolEncoding,
        dtype: DType,
        len: usize,
        _metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        buffers: &[ByteBuffer],
        children: &[ArrayParts],
        ctx: &ArrayContext,
    ) -> VortexResult<ByteBoolArray> {
        let validity = if children.is_empty() {
            Validity::from(dtype.nullability())
        } else if children.len() == 1 {
            let validity = children[0].decode(ctx, Validity::DTYPE, len)?;
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

    fn with_children(array: &ByteBoolArray, children: &[ArrayRef]) -> VortexResult<ByteBoolArray> {
        let validity = if array.validity().is_array() {
            Validity::Array(children[0].clone())
        } else {
            array.validity().clone()
        };

        Ok(ByteBoolArray::new(array.buffer().clone(), validity))
    }
}
