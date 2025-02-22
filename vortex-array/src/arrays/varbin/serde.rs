use std::fmt::{Debug, Display};

use serde::{Deserialize, Serialize};
use vortex_dtype::{DType, PType};
use vortex_error::VortexResult;

use crate::arrays::{VarBinArray, VarBinEncoding};
use crate::serde::ArrayParts;
use crate::validity::ValidityMetadata;
use crate::vtable::SerdeVTable;
use crate::{
    Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayRef, ArrayVisitorImpl, ContextRef,
    RkyvMetadata,
};

#[derive(Debug, Clone, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct VarBinMetadata {
    pub(crate) offsets_ptype: PType,
    pub(crate) bytes_len: usize,
}

impl ArrayVisitorImpl<RkyvMetadata<VarBinMetadata>> for VarBinArray {
    fn _buffers(&self, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(self.bytes()); // TODO(ngates): sliced bytes?
    }

    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("offsets", self.offsets())?;
        visitor.visit_validity(self.validity(), self.len());
    }

    fn _metadata(&self) -> RkyvMetadata<VarBinMetadata> {
        RkyvMetadata(VarBinMetadata {
            offsets_ptype: self.offsets().dtype().ptype(),
            bytes_len: self.bytes().len(),
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
        todo!()
    }
}
