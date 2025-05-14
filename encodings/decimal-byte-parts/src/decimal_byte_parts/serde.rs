use vortex_array::serde::ArrayChildren;
use vortex_array::vtable::{SerdeVTable, VisitorVTable};
use vortex_array::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, DeserializeMetadata, ProstMetadata,
};
use vortex_buffer::ByteBuffer;
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::PType::U64;
use vortex_dtype::{DType, PType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use crate::{DecimalBytePartsArray, DecimalBytePartsEncoding, DecimalBytePartsVTable};

#[derive(Clone, prost::Message)]
pub struct DecimalBytesPartsMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    zeroth_child_ptype: i32,
    #[prost(uint32, tag = "2")]
    lower_part_count: u32,
}

impl SerdeVTable<DecimalBytePartsVTable> for DecimalBytePartsVTable {
    type Metadata = ProstMetadata<DecimalBytesPartsMetadata>;

    fn metadata(array: &DecimalBytePartsArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(ProstMetadata(DecimalBytesPartsMetadata {
            zeroth_child_ptype: PType::try_from(array.msp.dtype()).vortex_expect("must be a PType")
                as i32,
            lower_part_count: u32::try_from(array.lower_parts.len())
                .vortex_expect("1..=3 fits in u8"),
        })))
    }

    fn build(
        _encoding: &DecimalBytePartsEncoding,
        dtype: &DType,
        len: usize,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<DecimalBytePartsArray> {
        let Some(decimal_dtype) = dtype.as_decimal() else {
            vortex_bail!("decoding decimal but given non decimal dtype {}", dtype)
        };

        let encoded_dtype = DType::Primitive(metadata.zeroth_child_ptype(), dtype.nullability());

        let msp = children.get(0, &encoded_dtype, len)?;

        let mut lower_parts = Vec::with_capacity(metadata.lower_part_count as usize);
        for idx in 0..metadata.lower_part_count {
            lower_parts.push(children.get(
                (idx + 1) as usize,
                &DType::Primitive(U64, NonNullable),
                len,
            )?)
        }

        DecimalBytePartsArray::try_new(msp, lower_parts, *decimal_dtype)
    }
}

const ENCODED_NAMES: [&str; 3] = ["lower-0", "lower-1", "lower-2"];

impl VisitorVTable<DecimalBytePartsVTable> for DecimalBytePartsVTable {
    fn visit_buffers(_array: &DecimalBytePartsArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &DecimalBytePartsArray, visitor: &mut dyn ArrayChildVisitor) {
        assert!(array.lower_parts.len() <= 3);
        visitor.visit_child("msp", &array.msp);
        array.lower_parts.iter().enumerate().for_each(|(idx, arr)| {
            visitor.visit_child(ENCODED_NAMES[idx], arr);
        })
    }
}
