use vortex_array::serde::ArrayChildren;
use vortex_array::vtable::{EncodeVTable, SerdeVTable, VisitorVTable};
use vortex_array::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayRef, Canonical, DeserializeMetadata,
    EmptyMetadata,
};
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, PType};
use vortex_error::{VortexResult, vortex_bail};

use crate::{ZigZagArray, ZigZagEncoding, ZigZagVTable, zigzag_encode};

impl SerdeVTable<ZigZagVTable> for ZigZagVTable {
    type Metadata = EmptyMetadata;

    fn metadata(_array: &ZigZagArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(EmptyMetadata))
    }

    fn build(
        _encoding: &ZigZagEncoding,
        dtype: &DType,
        len: usize,
        _metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ZigZagArray> {
        if children.len() != 1 {
            vortex_bail!("Expected 1 child, got {}", children.len());
        }

        let ptype = PType::try_from(dtype)?;
        let encoded_type = DType::Primitive(ptype.to_unsigned(), dtype.nullability());

        let encoded = children.get(0, &encoded_type, len)?;
        ZigZagArray::try_new(encoded)
    }
}

impl EncodeVTable<ZigZagVTable> for ZigZagVTable {
    fn encode(
        encoding: &ZigZagEncoding,
        canonical: &Canonical,
        _like: Option<&dyn Array>,
    ) -> VortexResult<Option<ZigZagArray>> {
        let parray = canonical.clone().into_primitive()?;

        if !parray.ptype().is_signed_int() {
            vortex_bail!(
                "only signed integers can be encoded into {}, got {}",
                encoding.id(),
                parray.ptype()
            )
        }

        Ok(Some(zigzag_encode(parray)?))
    }
}

impl VisitorVTable<ZigZagVTable> for ZigZagVTable {
    fn visit_buffers(_array: &ZigZagArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &ZigZagArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("encoded", array.encoded())
    }

    fn with_children(_array: &ZigZagArray, children: &[ArrayRef]) -> VortexResult<ZigZagArray> {
        let encoded = children[0].clone();
        ZigZagArray::try_new(encoded)
    }
}
