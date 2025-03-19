use vortex_buffer::ByteBufferMut;

use crate::arrays::ConstantArray;
use crate::{ArrayBufferVisitor, ArrayVisitorImpl, EmptyMetadata};

impl ArrayVisitorImpl for ConstantArray {
    fn _buffers(&self, visitor: &mut dyn ArrayBufferVisitor) {
        let buffer = self.scalar.value().to_flexbytes::<ByteBufferMut>().freeze();
        visitor.visit_buffer(&buffer);
    }

    fn _metadata(&self) -> EmptyMetadata {
        EmptyMetadata
    }
}
