use arrow_buffer::BooleanBuffer;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use super::BoolArray;
use crate::arrays::BoolVTable;
use crate::serde::ArrayParts;
use crate::validity::Validity;
use crate::vtable::{SerdeVTable, VTable, ValidityHelper, VisitorVTable};
use crate::{ArrayBufferVisitor, ArrayChildVisitor, ArrayContext, ArrayRef, ProstMetadata};

#[derive(prost::Message)]
pub struct BoolMetadata {
    // The offset in bits must be <8
    #[prost(uint32, tag = "1")]
    pub offset: u32,
}

impl SerdeVTable<BoolVTable> for BoolVTable {
    type Metadata = ProstMetadata<BoolMetadata>;

    fn metadata(array: &BoolArray) -> VortexResult<Option<Self::Metadata>> {
        let bit_offset = array.boolean_buffer().offset();
        assert!(bit_offset < 8, "Offset must be <8, got {}", bit_offset);
        Ok(Some(ProstMetadata(BoolMetadata {
            offset: u32::try_from(bit_offset).vortex_expect("checked"),
        })))
    }

    fn build(
        _encoding: &<BoolVTable as VTable>::Encoding,
        dtype: DType,
        len: usize,
        metadata: &BoolMetadata,
        buffers: &[ByteBuffer],
        children: &[ArrayParts],
        ctx: &ArrayContext,
    ) -> VortexResult<BoolArray> {
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
    fn visit_buffers(array: &BoolArray, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(&ByteBuffer::from_arrow_buffer(
            array.boolean_buffer().clone().into_inner(),
            Alignment::none(),
        ))
    }

    fn visit_children(array: &BoolArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(&array.validity, array.len());
    }

    fn with_children(array: &BoolArray, children: &[ArrayRef]) -> VortexResult<BoolArray> {
        let validity = if array.validity().is_array() {
            Validity::Array(children[0].clone())
        } else {
            array.validity().clone()
        };
        Ok(BoolArray::new(array.boolean_buffer().clone(), validity))
    }
}
