use vortex_array::{Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayVisitorImpl, EmptyMetadata};

use crate::ByteBoolArray;

impl ArrayVisitorImpl<EmptyMetadata> for ByteBoolArray {
    fn _buffers(&self, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(self.buffer());
    }

    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(self.validity(), self.len());
    }

    fn _metadata(&self) -> EmptyMetadata {
        EmptyMetadata
    }
}
