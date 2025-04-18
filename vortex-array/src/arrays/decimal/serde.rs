use vortex_buffer::Alignment;
use vortex_dtype::{DType, DecimalDType};
use vortex_error::{VortexResult, vortex_bail};

use super::{DecimalArray, DecimalEncoding};
use crate::serde::ArrayParts;
use crate::validity::Validity;
use crate::vtable::EncodingVTable;
use crate::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayContext, ArrayRef, ArrayVisitorImpl,
    Canonical, EmptyMetadata, EncodingId,
};

impl EncodingVTable for DecimalEncoding {
    fn id(&self) -> EncodingId {
        EncodingId::new_ref("vortex.decimal")
    }

    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ArrayContext,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        if parts.nbuffers() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", parts.nbuffers());
        }
        let buffer = parts.buffer(0)?;

        let validity = if parts.nchildren() == 0 {
            Validity::from(dtype.nullability())
        } else if parts.nchildren() == 1 {
            let validity = parts.child(0).decode(ctx, Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 0 or 1 child, got {}", parts.nchildren());
        };

        let decimal_dtype = match &dtype {
            DType::Decimal(decimal_dtype, _) => *decimal_dtype,
            _ => vortex_bail!("Expected Decimal dtype, got {:?}", dtype),
        };

        // Assuming 16-byte alignment for decimal values
        if !buffer.is_aligned(Alignment::new(16)) {
            vortex_bail!("Buffer is not aligned to 16-byte boundary");
        }
        if buffer.len() != 16 * len {
            vortex_bail!(
                "Buffer length {} does not match expected length {} for decimal values",
                buffer.len(),
                16 * len
            );
        }

        Ok(DecimalArray::new(buffer, decimal_dtype, validity).into_array())
    }

    fn encode(
        &self,
        input: &Canonical,
        _like: Option<&dyn Array>,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(input.clone().into_decimal()?.into_array()))
    }
}

impl ArrayVisitorImpl for DecimalArray {
    fn _visit_buffers(&self, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(self.byte_buffer());
    }

    fn _visit_children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(self.validity(), self.len());
    }

    fn _metadata(&self) -> EmptyMetadata {
        EmptyMetadata
    }
}