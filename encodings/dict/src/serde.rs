use vortex_array::serde::ArrayParts;
use vortex_array::vtable::SerdeVTable;
use vortex_array::{
    Array, ArrayChildVisitor, ArrayRef, ArrayVisitorImpl, ContextRef, DeserializeMetadata,
    RkyvMetadata,
};
use vortex_dtype::{DType, PType};
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

use crate::{DictArray, DictEncoding};

#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[repr(C)]
pub struct DictMetadata {
    codes_ptype: PType,
    values_len: u32,
}

impl ArrayVisitorImpl<RkyvMetadata<DictMetadata>> for DictArray {
    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("codes", self.codes());
        visitor.visit_child("values", self.values());
    }

    fn _metadata(&self) -> RkyvMetadata<DictMetadata> {
        RkyvMetadata(DictMetadata {
            codes_ptype: PType::try_from(self.codes().dtype())
                .vortex_expect("Must be a valid PType"),
            values_len: u32::try_from(self.values().len())
                .vortex_expect("Values length cannot exceed u32"),
        })
    }
}

impl SerdeVTable<&DictArray> for DictEncoding {
    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ContextRef,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        if parts.nchildren() != 2 {
            vortex_bail!(
                "Expected 2 children for dict encoding, found {}",
                parts.nchildren()
            )
        }
        let metadata = RkyvMetadata::<DictMetadata>::deserialize(parts.metadata())?;

        let codes_dtype = DType::Primitive(metadata.codes_ptype, dtype.nullability());
        let codes = parts.child(0).decode(ctx, codes_dtype, len)?;

        let values = parts
            .child(1)
            .decode(ctx, dtype, metadata.values_len as usize)?;

        Ok(DictArray::try_new(codes, values)?.into_array())
    }
}

#[cfg(test)]
mod test {
    use vortex_array::test_harness::check_metadata;
    use vortex_array::RkyvMetadata;
    use vortex_dtype::PType;

    use crate::serde::DictMetadata;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_dict_metadata() {
        check_metadata(
            "dict.metadata",
            RkyvMetadata(DictMetadata {
                codes_ptype: PType::U64,
                values_len: u32::MAX,
            }),
        );
    }
}
