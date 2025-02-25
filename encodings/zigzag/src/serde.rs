use vortex_array::serde::ArrayParts;
use vortex_array::vtable::SerdeVTable;
use vortex_array::{
    Array, ArrayChildVisitor, ArrayRef, ArrayVisitorImpl, ContextRef, EmptyMetadata,
};
use vortex_dtype::{DType, PType};
use vortex_error::{VortexResult, vortex_bail};

use crate::{ZigZagArray, ZigZagEncoding};

impl ArrayVisitorImpl for ZigZagArray {
    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("encoded", self.encoded())
    }

    fn _metadata(&self) -> EmptyMetadata {
        EmptyMetadata
    }
}

impl SerdeVTable<&ZigZagArray> for ZigZagEncoding {
    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ContextRef,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        if parts.nchildren() != 1 {
            vortex_bail!("Expected 1 child, got {}", parts.nchildren());
        }

        let ptype = PType::try_from(&dtype)?;
        let encoded_type = DType::Primitive(ptype.to_unsigned(), dtype.nullability());

        let encoded = parts.child(0).decode(ctx, encoded_type, len)?;
        Ok(ZigZagArray::try_new(encoded)?.into_array())
    }
}
