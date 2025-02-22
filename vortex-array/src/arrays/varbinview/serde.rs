use std::fmt::{Debug, Display, Formatter};

use vortex_error::VortexResult;

use crate::arrays::{VarBinViewArray, VarBinViewEncoding};
use crate::validity::ValidityMetadata;
use crate::vtable::SerdeVTable;
use crate::{Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayVisitorImpl, EmptyMetadata};

impl ArrayVisitorImpl<EmptyMetadata> for VarBinViewArray {
    fn _buffers(&self, visitor: &mut dyn ArrayBufferVisitor) {
        for buffer in self.buffers() {
            visitor.visit_buffer(buffer);
        }
    }

    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(self.validity(), self.len())
    }

    fn _metadata(&self) -> EmptyMetadata {
        EmptyMetadata
    }
}

impl SerdeVTable<&VarBinViewArray> for VarBinViewEncoding {}
