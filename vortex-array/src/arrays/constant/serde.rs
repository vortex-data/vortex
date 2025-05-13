use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};
use vortex_scalar::{Scalar, ScalarValue};

use crate::EmptyMetadata;
use crate::arrays::{ConstantArray, ConstantEncoding, ConstantVTable};
use crate::serde::ArrayChildren;
use crate::vtable::SerdeVTable;

impl SerdeVTable<ConstantVTable> for ConstantVTable {
    type Metadata = EmptyMetadata;

    fn metadata(_array: &ConstantArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(EmptyMetadata))
    }

    fn build(
        _encoding: &ConstantEncoding,
        dtype: &DType,
        len: usize,
        _metadata: &Self::Metadata,
        buffers: &[ByteBuffer],
        _children: &dyn ArrayChildren,
    ) -> VortexResult<ConstantArray> {
        if buffers.len() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", buffers.len());
        }
        let sv = ScalarValue::from_protobytes(&buffers[0])?;
        let scalar = Scalar::new(dtype.clone(), sv);
        Ok(ConstantArray::new(scalar, len))
    }
}
