use vortex_array::serde::ArrayParts;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::vtable::EncodingVTable;
use vortex_array::{
    Array, ArrayChildVisitor, ArrayContext, ArrayRef, ArrayVisitorImpl, Canonical, EmptyMetadata,
    EncodingId,
};
use vortex_dtype::{DType, PType};
use vortex_error::{VortexResult, vortex_bail};

use crate::{ZigZagArray, ZigZagEncoding, zigzag_encode};

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

    fn encode(
        &self,
        input: &Canonical,
        _like: Option<&dyn Array>,
    ) -> VortexResult<Option<ArrayRef>> {
        let parray = input.clone().into_primitive()?;

        if !parray.ptype().is_signed_int() {
            vortex_bail!(
                "only signed integers can be encoded into {}, got {}",
                self.id(),
                parray.ptype()
            )
        }

        Ok(Some(zigzag_encode(parray)?.into_array()))
    }
}

impl ArrayVisitorImpl for ZigZagArray {
    fn _visit_children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("encoded", self.encoded())
    }

    fn _metadata(&self) -> EmptyMetadata {
        EmptyMetadata
    }
}
