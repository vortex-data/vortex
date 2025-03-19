use itertools::Itertools;
use vortex_buffer::{Buffer, ByteBuffer};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use super::{BinaryView, VarBinViewEncoding};
use crate::arrays::VarBinViewArray;
use crate::serde::ArrayParts;
use crate::validity::Validity;
use crate::vtable::EncodingVTable;
use crate::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayContext, ArrayRef, ArrayVisitorImpl,
    EmptyMetadata, EncodingId,
};

impl EncodingVTable for VarBinViewEncoding {
    fn id(&self) -> EncodingId {
        EncodingId::new_ref("vortex.varbinview")
    }

    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ArrayContext,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        let mut buffers: Vec<ByteBuffer> = (0..parts.nbuffers())
            .map(|i| parts.buffer(i))
            .try_collect()?;
        let views = Buffer::<BinaryView>::from_byte_buffer(
            buffers.pop().vortex_expect("Missing views buffer"),
        );

        if views.len() != len {
            vortex_bail!("Expected {} views, got {}", len, views.len());
        }

        let validity = if parts.nchildren() == 0 {
            Validity::from(dtype.nullability())
        } else if parts.nchildren() == 1 {
            let validity = parts.child(0).decode(ctx, Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 0 or 1 children, got {}", parts.nchildren());
        };

        Ok(VarBinViewArray::try_new(views, buffers, dtype, validity)?.into_array())
    }
}

impl ArrayVisitorImpl<EmptyMetadata> for VarBinViewArray {
    fn _buffers(&self, visitor: &mut dyn ArrayBufferVisitor) {
        for buffer in self.buffers() {
            visitor.visit_buffer(buffer);
        }
        visitor.visit_buffer(&self.views().clone().into_byte_buffer());
    }

    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(self.validity(), self.len())
    }

    fn _metadata(&self) -> EmptyMetadata {
        EmptyMetadata
    }
}
