use crate::arrays::PrimitiveArray;
use crate::{Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayVisitorImpl, EmptyMetadata};

impl ArrayVisitorImpl for PrimitiveArray {
    fn _buffers(&self, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(self.byte_buffer());
    }

    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(self.validity(), self.len());
    }

    fn _metadata(&self) -> EmptyMetadata {
        EmptyMetadata
    }
}
