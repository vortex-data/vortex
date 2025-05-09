use vortex_buffer::{ByteBuffer, ByteBufferMut};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};
use vortex_scalar::{Scalar, ScalarValue};

use crate::arrays::{ConstantArray, ConstantVTable};
use crate::serde::ArrayParts;
use crate::vtable::SerdeVTable;
use crate::{
    Array, ArrayBufferVisitor, ArrayContext, ArrayRef, ArrayVisitorImpl, Canonical, EmptyMetadata,
};

impl SerdeVTable<ConstantVTable> for ConstantVTable {
    type Metadata = EmptyMetadata;

    fn metadata(_array: &ConstantVTable::Array) -> Self::Metadata {
        EmptyMetadata
    }

    fn decode2(
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

    fn encode2(
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

    fn encode(
        encoding: &ConstantVTable::Encoding,
        canonical: Canonical,
        like: Option<&dyn Array>,
    ) -> VortexResult<ConstantVTable::Array> {
        todo!()
    }

    fn decode(
        encoding: &ConstantVTable::Encoding,
        dtype: DType,
        len: usize,
        metadata: &Self::Metadata,
        buffers: &[ByteBuffer],
        children: &[ArrayParts],
        ctx: &ArrayContext,
    ) -> VortexResult<ConstantVTable::Array> {
        todo!()
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
