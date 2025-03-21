use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use super::ExtensionEncoding;
use crate::arrays::ExtensionArray;
use crate::serde::ArrayParts;
use crate::vtable::EncodingVTable;
use crate::{
    Array, ArrayChildVisitor, ArrayContext, ArrayRef, ArrayVisitorImpl, EmptyMetadata, EncodingId,
};

impl EncodingVTable for ExtensionEncoding {
    fn id(&self) -> EncodingId {
        EncodingId::new_ref("vortex.ext")
    }

    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ArrayContext,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        let DType::Extension(ext_dtype) = dtype else {
            vortex_bail!("Not an extension DType");
        };
        let storage = parts
            .child(0)
            .decode(ctx, ext_dtype.storage_dtype().clone(), len)?;
        Ok(ExtensionArray::new(ext_dtype, storage).into_array())
    }
}

impl ArrayVisitorImpl for ExtensionArray {
    fn _visit_children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("storage", self.storage())
    }

    fn _metadata(&self) -> EmptyMetadata {
        EmptyMetadata
    }
}
