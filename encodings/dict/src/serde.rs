use vortex_array::serde::ArrayChildren;
use vortex_array::vtable::{EncodeVTable, SerdeVTable, VisitorVTable};
use vortex_array::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, Canonical, DeserializeMetadata, ProstMetadata,
};
use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, PType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};

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

    fn metadata(array: &DictArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(ProstMetadata(DictMetadata {
            codes_ptype: PType::try_from(array.codes().dtype())
                .vortex_expect("Must be a valid PType") as i32,
            values_len: u32::try_from(array.values().len()).map_err(|_| {
                vortex_err!("Diction values cannot exceed u32 in length for serialization")
            })?,
        })))
    }

    fn build(
        _encoding: &DictEncoding,
        dtype: &DType,
        len: usize,
        metadata: &<Self::Metadata as DeserializeMetadata>::Output,
        _buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<DictArray> {
        if children.len() != 2 {
            vortex_bail!(
                "Expected 2 children for dict encoding, found {}",
                children.len()
            )
        }
        let codes_dtype = DType::Primitive(metadata.codes_ptype(), dtype.nullability());
        let codes = children.get(0, &codes_dtype, len)?;

        let values = children.get(1, dtype, metadata.values_len as usize)?;

        DictArray::try_new(codes, values)
    }
}

impl EncodeVTable<DictVTable> for DictVTable {
    fn encode(
        _encoding: &DictEncoding,
        canonical: &Canonical,
        _like: Option<&DictArray>,
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
