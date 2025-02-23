use std::fmt::{Debug, Display};

use serde::{Deserialize, Serialize};
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

use crate::arrays::{VarBinArray, VarBinEncoding};
use crate::serde::ArrayParts;
use crate::validity::Validity;
use crate::vtable::SerdeVTable;
use crate::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayRef, ArrayVisitorImpl, ContextRef,
    DeserializeMetadata, RkyvMetadata,
};

#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct VarBinMetadata {
    pub(crate) offsets_ptype: PType,
}

impl ArrayVisitorImpl<RkyvMetadata<VarBinMetadata>> for VarBinArray {
    fn _buffers(&self, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(self.bytes()); // TODO(ngates): sliced bytes?
    }

    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("offsets", self.offsets());
        visitor.visit_validity(self.validity(), self.len());
    }

    fn _metadata(&self) -> RkyvMetadata<VarBinMetadata> {
        RkyvMetadata(VarBinMetadata {
            offsets_ptype: PType::try_from(self.offsets().dtype())
                .vortex_expect("Must be a valid PType"),
        })
    }
}

impl SerdeVTable<&VarBinArray> for VarBinEncoding {
    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ContextRef,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        let metadata = RkyvMetadata::<VarBinMetadata>::deserialize(parts.metadata())?;

        let validity = if parts.nchildren() == 1 {
            Validity::from(dtype.nullability())
        } else if parts.nchildren() == 2 {
            let validity = parts.child(1).decode(ctx, Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 1 or 2 children, got {}", parts.nchildren());
        };

        let offsets = parts.child(0).decode(
            ctx,
            DType::Primitive(metadata.offsets_ptype, Nullability::NonNullable),
            len + 1,
        )?;

        if parts.nbuffers() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", parts.nbuffers());
        }
        let bytes = parts.buffer(0)?;

        Ok(VarBinArray::try_new(offsets, bytes, dtype, validity)?.into_array())
    }
}
