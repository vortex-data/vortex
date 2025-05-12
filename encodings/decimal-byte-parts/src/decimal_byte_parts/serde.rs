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

        let msp = parts.child(0).decode(ctx, encoded_dtype, len)?;

        let mut lower_parts = Vec::with_capacity(metadata.lower_part_count as usize);
        for idx in 0..metadata.lower_part_count {
            lower_parts.push(parts.child((idx + 1) as usize).decode(
                ctx,
                DType::Primitive(U64, NonNullable),
                len,
            )?)
        }

        DecimalBytePartsArray::try_new(msp, lower_parts, *decimal_dtype).map(|d| d.to_array())
    }
}

#[derive(Clone, prost::Message)]
pub struct DecimalBytesPartsMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    zeroth_child_ptype: i32,
    #[prost(uint32, tag = "2")]
    lower_part_count: u32,
}

const ENCODED_NAMES: [&str; 3] = ["lower-0", "lower-1", "lower-2"];

impl ArrayVisitorImpl<ProstMetadata<DecimalBytesPartsMetadata>> for DecimalBytePartsArray {
    fn _visit_children(&self, visitor: &mut dyn ArrayChildVisitor) {
        assert!(self.lower_parts.len() <= 3);
        visitor.visit_child("msp", &self.msp);
        self.lower_parts.iter().enumerate().for_each(|(idx, arr)| {
            visitor.visit_child(ENCODED_NAMES[idx], arr);
        })
    }

    fn _metadata(&self) -> ProstMetadata<DecimalBytesPartsMetadata> {
        ProstMetadata(DecimalBytesPartsMetadata {
            zeroth_child_ptype: PType::try_from(self.msp.dtype()).vortex_expect("must be a PType")
                as i32,
            lower_part_count: u32::try_from(self.lower_parts.len())
                .vortex_expect("1..=3 fits in u8"),
        })
    }
}
