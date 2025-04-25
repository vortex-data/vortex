use vortex_array::serde::ArrayParts;
use vortex_array::vtable::EncodingVTable;
use vortex_array::{
    Array, ArrayChildVisitor, ArrayContext, ArrayRef, ArrayVisitorImpl, Canonical,
    DeserializeMetadata, EncodingId, ProstMetadata,
};
use vortex_dtype::{DType, PType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use crate::builders::dict_encode;
use crate::{DictArray, DictEncoding};

#[derive(Clone, prost::Message)]
pub struct DictMetadata {
    #[prost(uint32, tag = "1")]
    values_len: u32,
    #[prost(enumeration = "PType", tag = "2")]
    codes_ptype: i32,
}

impl EncodingVTable for DictEncoding {
    fn id(&self) -> EncodingId {
        EncodingId::new_ref("vortex.dict")
    }

    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ArrayContext,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        if parts.nchildren() != 2 {
            vortex_bail!(
                "Expected 2 children for dict encoding, found {}",
                parts.nchildren()
            )
        }
        let metadata = ProstMetadata::<DictMetadata>::deserialize(parts.metadata())?;

        let codes_dtype = DType::Primitive(metadata.codes_ptype(), dtype.nullability());
        let codes = parts.child(0).decode(ctx, codes_dtype, len)?;

        let values = parts
            .child(1)
            .decode(ctx, dtype, metadata.values_len as usize)?;

        Ok(DictArray::try_new(codes, values)?.into_array())
    }

    fn encode(
        &self,
        input: &Canonical,
        _like: Option<&dyn Array>,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(dict_encode(input.as_ref())?.into_array()))
    }
}

impl ArrayVisitorImpl<ProstMetadata<DictMetadata>> for DictArray {
    fn _visit_children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("codes", self.codes());
        visitor.visit_child("values", self.values());
    }

    fn _metadata(&self) -> ProstMetadata<DictMetadata> {
        ProstMetadata(DictMetadata {
            codes_ptype: PType::try_from(self.codes().dtype())
                .vortex_expect("Must be a valid PType") as i32,
            values_len: u32::try_from(self.values().len())
                .vortex_expect("Values length cannot exceed u32"),
        })
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
