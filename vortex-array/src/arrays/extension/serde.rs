use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use super::ExtensionEncoding;
use crate::EmptyMetadata;
use crate::arrays::{ExtensionArray, ExtensionVTable};
use crate::serde::ArrayChildren;
use crate::vtable::SerdeVTable;

impl SerdeVTable<ExtensionVTable> for ExtensionVTable {
    type Metadata = EmptyMetadata;

    fn metadata(_array: &ExtensionArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(EmptyMetadata))
    }

    fn build(
        _encoding: &ExtensionEncoding,
        dtype: &DType,
        len: usize,
        _metadata: &Self::Metadata,
        _buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ExtensionArray> {
        let DType::Extension(ext_dtype) = dtype else {
            vortex_bail!("Not an extension DType");
        };
        if children.len() != 1 {
            vortex_bail!("Expected 1 child, got {}", children.len());
        }
        let storage = children.get(0, ext_dtype.storage_dtype(), len)?;
        Ok(ExtensionArray::new(ext_dtype.clone(), storage))
    }
}
