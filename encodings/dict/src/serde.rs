use vortex_array::serde::ArrayParts;
use vortex_array::vtable::{EncodeVTable, SerdeVTable, VisitorVTable};
use vortex_array::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayContext, ArrayRef, Canonical,
    DeserializeMetadata, ProstMetadata,
};
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, PType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use crate::builders::dict_encode;
use crate::{DictArray, DictEncoding, DictVTable};

#[derive(Clone, prost::Message)]
pub struct DictMetadata {
    #[prost(uint32, tag = "1")]
    values_len: u32,
    #[prost(enumeration = "PType", tag = "2")]
    codes_ptype: i32,
}

impl SerdeVTable<DictVTable> for DictVTable {
    type Metadata = ProstMetadata<DictMetadata>;

    fn metadata(array: &DictArray) -> Option<Self::Metadata> {
        Some(ProstMetadata(DictMetadata {
            codes_ptype: PType::try_from(array.codes().dtype())
                .vortex_expect("Must be a valid PType") as i32,
            values_len: u32::try_from(array.values().len())
                .vortex_expect("Values length cannot exceed u32"),
        }))
    }

    fn decode(
        _encoding: &DictEncoding,
        dtype: DType,
        len: usize,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _buffers: &[ByteBuffer],
        children: &[ArrayParts],
        ctx: &ArrayContext,
    ) -> VortexResult<DictArray> {
        if children.len() != 2 {
            vortex_bail!(
                "Expected 2 children for dict encoding, found {}",
                children.len()
            )
        }
        let codes_dtype = DType::Primitive(metadata.codes_ptype(), dtype.nullability());
        let codes = children[0].decode(ctx, codes_dtype, len)?;

        let values = children[1].decode(ctx, dtype, metadata.values_len as usize)?;

        DictArray::try_new(codes, values)
    }
}

impl EncodeVTable<DictVTable> for DictVTable {
    fn encode(
        _encoding: &DictEncoding,
        canonical: &Canonical,
        _like: Option<&dyn Array>,
    ) -> VortexResult<Option<DictArray>> {
        Ok(Some(dict_encode(canonical.as_ref())?))
    }
}

impl VisitorVTable<DictVTable> for DictVTable {
    fn visit_buffers(_array: &DictArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &DictArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("codes", array.codes());
        visitor.visit_child("values", array.values());
    }

    fn with_children(_array: &DictArray, children: &[ArrayRef]) -> VortexResult<DictArray> {
        let codes = children[0].clone();
        let values = children[1].clone();
        DictArray::try_new(codes, values)
    }
}

#[cfg(test)]
mod test {
    use vortex_array::ProstMetadata;
    use vortex_array::test_harness::check_metadata;
    use vortex_dtype::PType;

    use crate::serde::DictMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_dict_metadata() {
        check_metadata(
            "dict.metadata",
            ProstMetadata(DictMetadata {
                codes_ptype: PType::U64 as i32,
                values_len: u32::MAX,
            }),
        );
    }
}
