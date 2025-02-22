use arrow_buffer::BooleanBuffer;
use vortex_dtype::DType;
use vortex_error::{vortex_bail, VortexResult};

use crate::arrays::{BoolArray, BoolEncoding};
use crate::serde::ArrayParts;
use crate::validity::Validity;
use crate::vtable::SerdeVTable;
use crate::{Array, ArrayRef, ContextRef};

impl SerdeVTable<&BoolArray> for BoolEncoding {
    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: ContextRef,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        if parts.nbuffers() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", parts.nbuffers());
        }
        let buffer = BooleanBuffer::new(parts.buffers()?[0].clone().into_arrow_buffer(), 0, len);

        let validity = if parts.nchildren() == 0 {
            Validity::from(dtype.nullability())
        } else if parts.nchildren() == 1 {
            let validity = parts.child(0).decode(ctx, Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 0 or 1 child, got {}", parts.nchildren());
        };

        Ok(BoolArray::new(buffer, validity).into_array())
    }
}
