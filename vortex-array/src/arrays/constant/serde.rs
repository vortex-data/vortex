use vortex_buffer::ByteBufferMut;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};
use vortex_scalar::{Scalar, ScalarValue};

use super::ConstantEncoding;
use crate::arrays::ConstantArray;
use crate::serde::ArrayParts;
use crate::vtable::EncodingVTable;
use crate::{
    Array, ArrayBufferVisitor, ArrayContext, ArrayRef, ArrayVisitorImpl, EmptyMetadata, EncodingId,
};

impl EncodingVTable for ConstantEncoding {
    fn id(&self) -> EncodingId {
        EncodingId::new_ref("vortex.constant")
    }

    fn decode(
        &self,
        parts: &ArrayParts,
        _ctx: &ArrayContext,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        if parts.nbuffers() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", parts.nbuffers());
        }
        let sv = ScalarValue::from_protobytes(&parts.buffer(0)?)?;
        let scalar = Scalar::new(dtype, sv);
        Ok(ConstantArray::new(scalar, len).into_array())
    }

    fn encode(
        &self,
        input: &crate::Canonical,
        _like: Option<&dyn Array>,
    ) -> VortexResult<Option<ArrayRef>> {
        let array_ref = input.as_ref();
        if array_ref.is_constant() {
            let scalar = array_ref.scalar_at(0)?;
            Ok(Some(
                ConstantArray::new(scalar, array_ref.len()).into_array(),
            ))
        } else {
            Ok(None)
        }
    }
}

impl ArrayVisitorImpl for ConstantArray {
    fn _visit_buffers(&self, visitor: &mut dyn ArrayBufferVisitor) {
        let buffer = self
            .scalar
            .value()
            .to_protobytes::<ByteBufferMut>()
            .freeze();
        visitor.visit_buffer(&buffer);
    }

    fn _metadata(&self) -> EmptyMetadata {
        EmptyMetadata
    }
}
