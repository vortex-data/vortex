use arrow_buffer::BooleanBuffer;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use super::BoolArray;
use crate::arrays::BoolVTable;
use crate::serde::ArrayParts;
use crate::validity::Validity;
use crate::vtable::{SerdeVTable, VTable, VisitorVTable};
use crate::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayContext, Canonical, Encoding, ProstMetadata,
};

#[derive(prost::Message)]
pub struct BoolMetadata {
    // The offset in bits must be <8
    #[prost(uint32, tag = "1")]
    pub offset: u32,
}

impl SerdeVTable<BoolVTable> for BoolVTable {
    type Metadata = ProstMetadata<BoolMetadata>;

    fn metadata(array: &<BoolVTable as VTable>::Array) -> Self::Metadata {
        let bit_offset = array.boolean_buffer().offset();
        assert!(bit_offset < 8, "Offset must be <8, got {}", bit_offset);
        ProstMetadata(BoolMetadata {
            offset: u32::try_from(bit_offset).vortex_expect("checked"),
        })
    }

    fn encode(
        _encoding: &<BoolVTable as VTable>::Encoding,
        _canonical: Canonical,
        _like: Option<&dyn Array>,
    ) -> VortexResult<<BoolVTable as VTable>::Array> {
        todo!()
    }

    fn decode(
        _encoding: &<BoolVTable as VTable>::Encoding,
        dtype: DType,
        len: usize,
        metadata: &Self::Metadata,
        buffers: &[ByteBuffer],
        children: &[ArrayParts],
        ctx: &ArrayContext,
    ) -> VortexResult<<BoolVTable as VTable>::Array> {
        if buffers.len() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", buffers.len());
        }
        let buffer = BooleanBuffer::new(
            buffers[0].clone().into_arrow_buffer(),
            metadata.offset as usize,
            len,
        );

        let validity = if children.is_empty() {
            Validity::from(dtype.nullability())
        } else if children.len() == 1 {
            let validity = children[0].decode(ctx, Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 0 or 1 child, got {}", children.len());
        };

        Ok(BoolArray::new(buffer, validity))
    }
}

impl VisitorVTable<BoolVTable> for BoolVTable {
    fn visit_buffers(array: &<BoolVTable as VTable>::Array, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(&ByteBuffer::from_arrow_buffer(
            array.boolean_buffer().clone().into_inner(),
            Alignment::none(),
        ))
    }

    fn visit_children(array: &<BoolVTable as VTable>::Array, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(&array.validity, array.len());
    }
}
