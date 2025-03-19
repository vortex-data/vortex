use crate::arrays::VarBinViewArray;
use crate::{Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayVisitorImpl, EmptyMetadata};

impl ArrayVisitorImpl<EmptyMetadata> for VarBinViewArray {
    fn _buffers(&self, visitor: &mut dyn ArrayBufferVisitor) {
        for buffer in self.buffers() {
            visitor.visit_buffer(buffer);
        }
        visitor.visit_buffer(&self.views().clone().into_byte_buffer());
    }

    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(self.validity(), self.len())
    }

    fn _metadata(&self) -> EmptyMetadata {
        EmptyMetadata
    }
}
