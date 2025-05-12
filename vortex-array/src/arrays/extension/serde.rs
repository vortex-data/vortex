use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use super::ExtensionEncoding;
use crate::arrays::{ExtensionArray, ExtensionVTable};
use crate::serde::ArrayParts;
use crate::vtable::SerdeVTable;
use crate::{ArrayContext, EmptyMetadata};

impl SerdeVTable<ExtensionVTable> for ExtensionVTable {
    type Metadata = EmptyMetadata;

    fn metadata(_array: &ExtensionArray) -> Option<Self::Metadata> {
        Some(EmptyMetadata)
    }

    fn decode(
        _encoding: &ExtensionEncoding,
        dtype: DType,
        len: usize,
        _metadata: &Self::Metadata,
        _buffers: &[ByteBuffer],
        children: &[ArrayParts],
        ctx: &ArrayContext,
    ) -> VortexResult<ExtensionArray> {
        let DType::Extension(ext_dtype) = dtype else {
            vortex_bail!("Not an extension DType");
        };
        if children.len() != 1 {
            vortex_bail!("Expected 1 child, got {}", children.len());
        }
        let storage = children[0].decode(ctx, ext_dtype.storage_dtype().clone(), len)?;
        Ok(ExtensionArray::new(ext_dtype, storage))
    }
}
