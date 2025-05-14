use vortex_buffer::ByteBuffer;
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use super::VarBinEncoding;
use crate::arrays::{VarBinArray, VarBinVTable};
use crate::serde::ArrayParts;
use crate::validity::Validity;
use crate::vtable::{SerdeVTable, ValidityHelper, VisitorVTable};
use crate::{Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayContext, ArrayRef, ProstMetadata};

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
        dtype: DType,
        len: usize,
        metadata: &VarBinMetadata,
        buffers: &[ByteBuffer],
        children: &[ArrayParts],
        ctx: &ArrayContext,
    ) -> VortexResult<VarBinArray> {
        let validity = if children.len() == 1 {
            Validity::from(dtype.nullability())
        } else if children.len() == 2 {
            let validity = children[1].decode(ctx, Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 1 or 2 children, got {}", children.len());
        };

        let offsets = children[0].decode(
            ctx,
            DType::Primitive(metadata.offsets_ptype(), Nullability::NonNullable),
            len + 1,
        )?;

        if buffers.len() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", buffers.len());
        }
        let bytes = buffers[0].clone();

        VarBinArray::try_new(offsets, bytes, dtype, validity)
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

    fn with_children(array: &VarBinArray, children: &[ArrayRef]) -> VortexResult<VarBinArray> {
        let new = match children.len() {
            // Only the offsets array is mandatory
            1 => {
                let offsets = children[0].clone();
                VarBinArray::try_new(
                    offsets,
                    array.bytes().clone(),
                    array.dtype().clone(),
                    array.validity().clone(),
                )?
            }
            // If are provided with both an offsets and validity arrays
            2 => {
                let offsets = children[0].clone();
                let validity_array = children[1].clone();
                VarBinArray::try_new(
                    offsets,
                    array.bytes().clone(),
                    array.dtype().clone(),
                    Validity::Array(validity_array),
                )?
            }
            _ => vortex_bail!("unexpected number of new children"),
        };

        Ok(new)
    }
}
