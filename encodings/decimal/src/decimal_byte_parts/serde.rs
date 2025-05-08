use vortex_array::serde::ArrayParts;
use vortex_array::vtable::EncodingVTable;
use vortex_array::{
    Array, ArrayChildVisitor, ArrayContext, ArrayRef, ArrayVisitorImpl, DeserializeMetadata,
    EncodingId, ProstMetadata,
};
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::PType::U64;
use vortex_dtype::{DType, PType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use crate::{DecimalBytePartsArray, DecimalBytePartsEncoding};

impl EncodingVTable for DecimalBytePartsEncoding {
    fn id(&self) -> EncodingId {
        EncodingId::new_ref("vortex.decimal_bytes_parts")
    }

    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ArrayContext,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        let metadata = ProstMetadata::<DecimalBytesPartsMetadata>::deserialize(parts.metadata())?;

        let Some(decimal_dtype) = dtype.as_decimal() else {
            vortex_bail!("decoding decimal but given non decimal dtype {}", dtype)
        };

        let encoded_dtype = DType::Primitive(metadata.zeroth_child_ptype(), dtype.nullability());

        let mut encoded = Vec::with_capacity(metadata.child_count as usize);
        encoded.push(parts.child(0).decode(ctx, encoded_dtype, len)?);
        for idx in 1..metadata.child_count {
            encoded.push(parts.child(idx as usize).decode(
                ctx,
                DType::Primitive(U64, NonNullable),
                len,
            )?)
        }

        DecimalBytePartsArray::try_new(encoded, *decimal_dtype).map(|d| d.to_array())
    }
}

#[derive(Clone, prost::Message)]
pub struct DecimalBytesPartsMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    zeroth_child_ptype: i32,
    #[prost(uint32, tag = "2")]
    child_count: u32,
}

const ENCODED_NAMES: [&str; 4] = ["parts-0", "parts-1", "parts-2", "parts-3"];

impl ArrayVisitorImpl<ProstMetadata<DecimalBytesPartsMetadata>> for DecimalBytePartsArray {
    fn _visit_children(&self, visitor: &mut dyn ArrayChildVisitor) {
        assert!(self.parts.len() <= 4);
        self.parts.iter().enumerate().for_each(|(idx, arr)| {
            visitor.visit_child(ENCODED_NAMES[idx], arr);
        })
    }

    fn _metadata(&self) -> ProstMetadata<DecimalBytesPartsMetadata> {
        ProstMetadata(DecimalBytesPartsMetadata {
            zeroth_child_ptype: PType::try_from(self.parts[0].dtype())
                .vortex_expect("must be a PType") as i32,
            child_count: u32::try_from(self.parts.len()).vortex_expect("0..4 fits in u8"),
        })
    }
}
