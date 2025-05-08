use vortex_buffer::ByteBuffer;

use crate::vtable::VTable;
use crate::{Array, ArrayBufferVisitor, ArrayChildVisitor};

pub trait VisitorVTable<V: VTable> {
    fn visit_buffers(array: &V::Array, _visitor: &mut dyn ArrayBufferVisitor);

    fn nbuffers(array: &V::Array) -> usize {
        struct NBuffers(usize);

        impl ArrayBufferVisitor for NBuffers {
            fn visit_buffer(&mut self, _buffer: &ByteBuffer) {
                self.0 += 1;
            }
        }

        let mut visitor = NBuffers(0);
        array.visit_buffers(&mut visitor);
        visitor.0
    }

    fn visit_children(array: &V::Array, _visitor: &mut dyn ArrayChildVisitor);

    fn nchildren(array: &V::Array) -> usize {
        struct NChildren(usize);

        impl ArrayChildVisitor for NChildren {
            fn visit_child(&mut self, _name: &str, _array: &dyn Array) {
                self.0 += 1;
            }
        }

        let mut visitor = NChildren(0);
        array.visit_children(&mut visitor);
        visitor.0
    }
}
