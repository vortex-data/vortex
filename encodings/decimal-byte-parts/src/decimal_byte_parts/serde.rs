// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::serde::ArrayChildren;
use vortex_array::vtable::{SerdeVTable, VisitorVTable};
use vortex_array::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, DeserializeMetadata, ProstMetadata,
};
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, PType};
use vortex_error::{VortexResult, vortex_bail};

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
            zeroth_child_ptype: PType::try_from(array.msp.dtype())? as i32,
            lower_part_count: 0,
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
        let Some(decimal_dtype) = dtype.as_decimal_opt() else {
            vortex_bail!("decoding decimal but given non decimal dtype {}", dtype)
        };

        let encoded_dtype = DType::Primitive(metadata.zeroth_child_ptype(), dtype.nullability());

        let msp = children.get(0, &encoded_dtype, len)?;

        assert_eq!(
            metadata.lower_part_count, 0,
            "lower_part_count > 0 not currently supported"
        );

        DecimalBytePartsArray::try_new(msp, *decimal_dtype)
    }
}

impl VisitorVTable<DecimalBytePartsVTable> for DecimalBytePartsVTable {
    fn visit_buffers(_array: &DecimalBytePartsArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &DecimalBytePartsArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("msp", &array.msp);
    }
}
