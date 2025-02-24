use arrow_buffer::BooleanBuffer;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

use crate::arrays::{BoolArray, BoolEncoding};
use crate::serde::ArrayParts;
use crate::validity::Validity;
use crate::vtable::SerdeVTable;
use crate::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayRef, ArrayVisitorImpl, ContextRef,
    DeserializeMetadata, RkyvMetadata,
};

#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct BoolMetadata {
    // We know the offset in bits must be <8
    offset: u8,
}

impl ArrayVisitorImpl<RkyvMetadata<BoolMetadata>> for BoolArray {
    fn _buffers(&self, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(&ByteBuffer::from_arrow_buffer(
            self.boolean_buffer().clone().into_inner(),
            Alignment::none(),
        ))
    }

    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(&self.validity, self.len());
    }

    fn _metadata(&self) -> RkyvMetadata<BoolMetadata> {
        let bit_offset = self.boolean_buffer().offset();
        assert!(bit_offset < 8, "Offset must be <8, got {}", bit_offset);
        RkyvMetadata(BoolMetadata {
            offset: u8::try_from(bit_offset).vortex_expect("checked"),
        })
    }
}

impl SerdeVTable<&BoolArray> for BoolEncoding {
    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ContextRef,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        let metadata = RkyvMetadata::<BoolMetadata>::deserialize(parts.metadata())?;

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
