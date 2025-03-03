use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::arrays::{ExtensionArray, ExtensionEncoding};
use crate::serde::ArrayParts;
use crate::vtable::SerdeVTable;
use crate::{Array, ArrayChildVisitor, ArrayContext, ArrayRef, ArrayVisitorImpl, EmptyMetadata};

impl ArrayVisitorImpl for ExtensionArray {
    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("storage", self.storage())
    }

    fn _metadata(&self) -> EmptyMetadata {
        EmptyMetadata
    }
}

impl SerdeVTable<&ExtensionArray> for ExtensionEncoding {
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
