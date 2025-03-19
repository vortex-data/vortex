use vortex_array::serde::ArrayParts;
use vortex_array::validity::Validity;
use vortex_array::vtable::EncodingVTable;
use vortex_array::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayContext, ArrayRef, ArrayVisitorImpl,
    EmptyMetadata, EncodingId,
};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::{ByteBoolArray, ByteBoolEncoding};

impl EncodingVTable for ByteBoolEncoding {
    fn id(&self) -> EncodingId {
        EncodingId::new_ref("vortex.bytebool")
    }

    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ArrayContext,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        let validity = if parts.nchildren() == 0 {
            Validity::from(dtype.nullability())
        } else if parts.nchildren() == 1 {
            let validity = parts.child(0).decode(ctx, Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 0 or 1 child, got {}", parts.nchildren());
        };

        if parts.nbuffers() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", parts.nbuffers());
        }
        let buffer = parts.buffer(0)?;

        Ok(ByteBoolArray::new(buffer, validity).into_array())
    }
}

impl ArrayVisitorImpl<EmptyMetadata> for ByteBoolArray {
    fn _buffers(&self, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(self.buffer());
    }

    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(self.validity(), self.len());
    }

    fn _metadata(&self) -> EmptyMetadata {
        EmptyMetadata
    }
}
