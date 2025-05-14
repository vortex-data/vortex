use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use super::VarBinEncoding;
use crate::arrays::{VarBinArray, VarBinVTable};
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable::{SerdeVTable, ValidityHelper, VisitorVTable};
use crate::{Array, ArrayBufferVisitor, ArrayChildVisitor, ProstMetadata};

#[derive(Clone, prost::Message)]
pub struct VarBinMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    pub(crate) offsets_ptype: i32,
}

impl SerdeVTable<VarBinVTable> for VarBinVTable {
    type Metadata = ProstMetadata<VarBinMetadata>;

    fn metadata(array: &VarBinArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(ProstMetadata(VarBinMetadata {
            offsets_ptype: PType::try_from(array.offsets().dtype())
                .vortex_expect("Must be a valid PType") as i32,
        })))
    }

    fn build(
        _encoding: &VarBinEncoding,
        dtype: &DType,
        len: usize,
        metadata: &VarBinMetadata,
        buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<VarBinArray> {
        let validity = if children.len() == 1 {
            Validity::from(dtype.nullability())
        } else if children.len() == 2 {
            let validity = children.get(1, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 1 or 2 children, got {}", children.len());
        };

        let offsets = children.get(
            0,
            &DType::Primitive(metadata.offsets_ptype(), Nullability::NonNullable),
            len + 1,
        )?;

        if buffers.len() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", buffers.len());
        }
        let bytes = buffers[0].clone();

        VarBinArray::try_new(offsets, bytes, dtype.clone(), validity)
    }
}

impl VisitorVTable<VarBinVTable> for VarBinVTable {
    fn visit_buffers(array: &VarBinArray, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(array.bytes()); // TODO(ngates): sliced bytes?
    }

    fn visit_children(array: &VarBinArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("offsets", array.offsets());
        visitor.visit_validity(array.validity(), array.len());
    }
}
