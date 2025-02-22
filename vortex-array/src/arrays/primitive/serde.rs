use vortex_buffer::{Alignment, Buffer};
use vortex_dtype::{match_each_native_ptype, DType, Nullability, PType};
use vortex_error::{vortex_bail, VortexResult};

use crate::arrays::{PrimitiveArray, PrimitiveEncoding};
use crate::serde::ArrayParts;
use crate::validity::Validity;
use crate::vtable::SerdeVTable;
use crate::{Array, ArrayRef, ContextRef};

impl SerdeVTable<&PrimitiveArray> for PrimitiveEncoding {
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
        let buffer = parts.buffers()?[0].clone();

        let validity = if parts.nchildren() == 0 {
            Validity::from(dtype.nullability())
        } else if parts.nchildren() == 1 {
            let validity = parts.children()[0]
                .clone()
                .decode(ctx, Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 0 or 1 child, got {}", parts.nchildren());
        };

        let ptype = PType::try_from(&dtype)?;

        if !buffer.is_aligned(Alignment::new(ptype.byte_width())) {
            vortex_bail!(
                "Buffer is not aligned to {}-byte boundary",
                ptype.byte_width()
            );
        }

        match_each_native_ptype!(ptype, |$P| {
            let buffer = Buffer::<$P>::from_byte_buffer(buffer);
            Ok(PrimitiveArray::new(buffer, validity).into_array())
        })
    }
}
