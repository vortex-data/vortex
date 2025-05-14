use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use super::{ListArray, ListVTable};
use crate::arrays::ListEncoding;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable::{SerdeVTable, ValidityHelper, VisitorVTable};
use crate::{Array, ArrayBufferVisitor, ArrayChildVisitor, ProstMetadata};

#[derive(Clone, prost::Message)]
pub struct ListMetadata {
    #[prost(uint64, tag = "1")]
    elements_len: u64,
    #[prost(enumeration = "PType", tag = "2")]
    offset_ptype: i32,
}

impl SerdeVTable<ListVTable> for ListVTable {
    type Metadata = ProstMetadata<ListMetadata>;

    fn metadata(array: &ListArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(ProstMetadata(ListMetadata {
            elements_len: array.elements().len() as u64,
            offset_ptype: PType::try_from(array.offsets().dtype())
                .vortex_expect("Must be a valid PType") as i32,
        })))
    }

    fn build(
        _encoding: &ListEncoding,
        dtype: &DType,
        len: usize,
        metadata: &ListMetadata,
        _buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ListArray> {
        let validity = if children.len() == 2 {
            Validity::from(dtype.nullability())
        } else if children.len() == 3 {
            let validity = children.get(2, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 2 or 3 children, got {}", children.len());
        };

        let DType::List(element_dtype, _) = &dtype else {
            vortex_bail!("Expected List dtype, got {:?}", dtype);
        };
        let elements = children.get(
            0,
            element_dtype.as_ref(),
            usize::try_from(metadata.elements_len).vortex_expect("Too many elements"),
        )?;

        let offsets = children.get(
            1,
            &DType::Primitive(metadata.offset_ptype(), Nullability::NonNullable),
            len + 1,
        )?;

        ListArray::try_new(elements, offsets, validity)
    }
}

impl VisitorVTable<ListVTable> for ListVTable {
    fn visit_buffers(_array: &ListArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &ListArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("elements", array.elements());
        visitor.visit_child("offsets", array.offsets());
        visitor.visit_validity(array.validity(), array.len());
    }
}
