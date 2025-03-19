use vortex_array::serde::ArrayParts;
use vortex_array::vtable::EncodingVTable;
use vortex_array::{Array, ArrayContext, ArrayRef, EncodingId};
use vortex_dtype::{DType, PType};
use vortex_error::{VortexResult, vortex_bail};

use crate::{ZigZagArray, ZigZagEncoding};

impl EncodingVTable for ZigZagEncoding {
    fn id(&self) -> EncodingId {
        EncodingId::new_ref("vortex.zigzag")
    }

    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ArrayContext,
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
