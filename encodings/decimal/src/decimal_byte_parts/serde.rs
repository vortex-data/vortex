use vortex_array::serde::ArrayParts;
use vortex_array::vtable::EncodingVTable;
use vortex_array::{
    Array, ArrayChildVisitor, ArrayContext, ArrayRef, ArrayVisitorImpl, DeserializeMetadata,
    EncodingId, ProstMetadata,
};
use vortex_dtype::{DType, PType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use super::{DecimalWrapperArray, DecimalWrapperEncoding};

impl EncodingVTable for DecimalWrapperEncoding {
    fn id(&self) -> EncodingId {
        EncodingId::new_ref("vortex.decimal_wrapper")
    }

    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ArrayContext,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        let metadata = ProstMetadata::<DecimalWrapperMetadata>::deserialize(parts.metadata())?;

        let Some(decimal_dtype) = dtype.as_decimal() else {
            vortex_bail!("decoding decimal but given non decimal dtype {}", dtype)
        };

        let encoded_dtype = DType::Primitive(metadata.inner_ptype(), dtype.nullability());

        let encoded = parts.child(0).decode(ctx, encoded_dtype, len)?;

        DecimalWrapperArray::try_new(encoded, *decimal_dtype).map(|d| d.to_array())
    }
}

#[derive(Clone, prost::Message)]
pub struct DecimalWrapperMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    inner_ptype: i32,
}

impl ArrayVisitorImpl<ProstMetadata<DecimalWrapperMetadata>> for DecimalWrapperArray {
    fn _visit_children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("encoded", self.encoded.as_ref());
    }

    fn _metadata(&self) -> ProstMetadata<DecimalWrapperMetadata> {
        ProstMetadata(DecimalWrapperMetadata {
            inner_ptype: PType::try_from(self.encoded.dtype()).vortex_expect("must be a PType")
                as i32,
        })
    }
}
