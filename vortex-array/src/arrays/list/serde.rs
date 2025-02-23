use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

use crate::arrays::{ListArray, ListEncoding};
use crate::serde::ArrayParts;
use crate::validity::Validity;
use crate::vtable::SerdeVTable;
use crate::{
    Array, ArrayChildVisitor, ArrayRef, ArrayVisitorImpl, ContextRef, DeserializeMetadata,
    RkyvMetadata,
};

#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct ListMetadata {
    elements_len: usize,
    offset_ptype: PType,
}

impl ArrayVisitorImpl<RkyvMetadata<ListMetadata>> for ListArray {
    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("elements", self.elements());
        visitor.visit_child("offsets", self.offsets());
        visitor.visit_validity(self.validity(), self.len());
    }

    fn _metadata(&self) -> RkyvMetadata<ListMetadata> {
        RkyvMetadata(ListMetadata {
            elements_len: self.elements().len(),
            offset_ptype: PType::try_from(self.offsets().dtype())
                .vortex_expect("Must be a valid PType"),
        })
    }
}

impl SerdeVTable<&ListArray> for ListEncoding {
    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ContextRef,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        let metadata = RkyvMetadata::<ListMetadata>::deserialize(parts.metadata())?;

        let validity = if parts.nchildren() == 2 {
            Validity::from(dtype.nullability())
        } else if parts.nchildren() == 3 {
            let validity = parts.child(2).decode(ctx, Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 2 or 3 children, got {}", parts.nchildren());
        };

        let DType::List(element_dtype, _) = &dtype else {
            vortex_bail!("Expected List dtype, got {:?}", dtype);
        };
        let elements =
            parts
                .child(0)
                .decode(ctx, element_dtype.as_ref().clone(), metadata.elements_len)?;

        let offsets = parts.child(1).decode(
            ctx,
            DType::Primitive(metadata.offset_ptype, Nullability::NonNullable),
            len + 1,
        )?;

        Ok(ListArray::try_new(elements, offsets, validity)?.into_array())
    }
}
