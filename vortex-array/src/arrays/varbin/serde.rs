use std::fmt::Debug;

use vortex_dtype::PType;
use vortex_error::VortexExpect;

use crate::arrays::VarBinArray;
use crate::{Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayVisitorImpl, RkyvMetadata};

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
