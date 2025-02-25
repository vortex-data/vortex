use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};
use vortex_scalar::{Scalar, ScalarValue};

use crate::arrays::{ConstantArray, ConstantEncoding};
use crate::serde::ArrayParts;
use crate::vtable::SerdeVTable;
use crate::{Array, ArrayBufferVisitor, ArrayRef, ArrayVisitorImpl, ContextRef, EmptyMetadata};

impl ArrayVisitorImpl for ConstantArray {
    fn _buffers(&self, visitor: &mut dyn ArrayBufferVisitor) {
        let buffer = self.scalar.value().to_flexbytes().into_inner();
        visitor.visit_buffer(&buffer);
    }

    fn _metadata(&self) -> EmptyMetadata {
        EmptyMetadata
    }
}

impl SerdeVTable<&ConstantArray> for ConstantEncoding {
    fn decode(
        &self,
        parts: &ArrayParts,
        _ctx: &ContextRef,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        if parts.nbuffers() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", parts.nbuffers());
        }
        let sv = ScalarValue::from_flexbytes(&parts.buffer(0)?)?;
        let scalar = Scalar::new(dtype, sv);
        Ok(ConstantArray::new(scalar, len).into_array())
    }
}
