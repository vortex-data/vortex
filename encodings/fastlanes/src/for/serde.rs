use vortex_array::serde::ArrayParts;
use vortex_array::vtable::SerdeVTable;
use vortex_array::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayRef, ArrayVisitorImpl, ContextRef,
    EmptyMetadata,
};
use vortex_dtype::{DType, PType};
use vortex_error::{VortexResult, vortex_bail};
use vortex_scalar::{Scalar, ScalarValue};

use crate::{FoRArray, FoREncoding};

impl ArrayVisitorImpl for FoRArray {
    fn _buffers(&self, visitor: &mut dyn ArrayBufferVisitor) {
        let pvalue = self.reference_scalar().value().to_flexbytes();
        visitor.visit_buffer(pvalue.as_ref());
    }

    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("encoded", self.encoded())
    }

    fn _metadata(&self) -> EmptyMetadata {
        EmptyMetadata
    }
}

impl SerdeVTable<&FoRArray> for FoREncoding {
    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ContextRef,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        if parts.nchildren() != 1 {
            vortex_bail!(
                "Expected 1 child for FoR encoding, found {}",
                parts.nchildren()
            )
        }

        let ptype = PType::try_from(&dtype)?;
        let encoded_dtype = DType::Primitive(ptype.to_unsigned(), dtype.nullability());
        let encoded = parts.child(0).decode(ctx, encoded_dtype, len)?;

        if parts.nbuffers() != 1 {
            vortex_bail!("Expected 1 buffers, got {}", parts.nbuffers());
        }
        let reference = Scalar::new(dtype, ScalarValue::from_flexbytes(&parts.buffer(0)?)?);

        Ok(FoRArray::try_new(encoded, reference)?.into_array())
    }
}
