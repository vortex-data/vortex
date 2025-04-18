use arrow_buffer::BooleanBuffer;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use super::{BoolArray, BoolEncoding};
use crate::serde::ArrayParts;
use crate::validity::Validity;
use crate::vtable::EncodingVTable;
use crate::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayContext, ArrayRef, ArrayVisitorImpl,
    DeserializeMetadata, Encoding, EncodingId, ProstMetadata,
};

#[derive(prost::Message)]
pub struct BoolMetadata {
    // The offset in bits must be <8
    #[prost(uint32, tag = "1")]
    pub offset: u32,
}

impl EncodingVTable for BoolEncoding {
    fn id(&self) -> EncodingId {
        EncodingId::new_ref("vortex.bool")
    }

    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ArrayContext,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        let metadata = <Self as Encoding>::Metadata::deserialize(parts.metadata())?;

        if parts.nbuffers() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", parts.nbuffers());
        }
        let buffer = BooleanBuffer::new(
            parts.buffer(0)?.into_arrow_buffer(),
            metadata.offset as usize,
            len,
        );

        let validity = if parts.nchildren() == 0 {
            Validity::from(dtype.nullability())
        } else if parts.nchildren() == 1 {
            let validity = parts.child(0).decode(ctx, Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 0 or 1 child, got {}", parts.nchildren());
        };

        Ok(BoolArray::new(buffer, validity).into_array())
    }
}

impl ArrayVisitorImpl<ProstMetadata<BoolMetadata>> for BoolArray {
    fn _visit_buffers(&self, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(&ByteBuffer::from_arrow_buffer(
            self.boolean_buffer().clone().into_inner(),
            Alignment::none(),
        ))
    }

    fn _visit_children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(&self.validity, self.len());
    }

    fn _metadata(&self) -> ProstMetadata<BoolMetadata> {
        let bit_offset = self.boolean_buffer().offset();
        assert!(bit_offset < 8, "Offset must be <8, got {}", bit_offset);
        ProstMetadata(BoolMetadata {
            offset: u32::try_from(bit_offset).vortex_expect("checked"),
        })
    }
}
