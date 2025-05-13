use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};
use vortex_scalar::{Scalar, ScalarValue};

use crate::arrays::{ConstantArray, ConstantEncoding, ConstantVTable};
use crate::serde::ArrayParts;
use crate::vtable::SerdeVTable;
use crate::{ArrayContext, EmptyMetadata};

impl SerdeVTable<ConstantVTable> for ConstantVTable {
    type Metadata = EmptyMetadata;

    fn metadata(_array: &ConstantArray) -> Option<Self::Metadata> {
        Some(EmptyMetadata)
    }

    fn decode(
        _encoding: &ConstantEncoding,
        dtype: DType,
        len: usize,
        _metadata: &Self::Metadata,
        buffers: &[ByteBuffer],
        _children: &[ArrayParts],
        _ctx: &ArrayContext,
    ) -> VortexResult<ConstantArray> {
        if buffers.len() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", buffers.len());
        }
        let sv = ScalarValue::from_protobytes(&buffers[0])?;
        let scalar = Scalar::new(dtype, sv);
        Ok(ConstantArray::new(scalar, len))
    }
}
