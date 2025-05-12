use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::vtable::VTable;
use crate::{Array, ArrayBufferVisitor, ArrayChildVisitor, ArrayRef};

pub trait VisitorVTable<V: VTable> {
    /// Visit the buffers of the array.
    fn visit_buffers(array: &V::Array, visitor: &mut dyn ArrayBufferVisitor);

    /// Count the number of buffers in the array.
    fn nbuffers(array: &V::Array) -> usize {
        struct NBuffers(usize);

        impl ArrayBufferVisitor for NBuffers {
            fn visit_buffer(&mut self, _buffer: &ByteBuffer) {
                self.0 += 1;
            }
        }

        let mut visitor = NBuffers(0);
        <V::VisitorVTable as VisitorVTable<V>>::visit_buffers(array, &mut visitor);
        visitor.0
    }

    /// Visit the children of the array.
    fn visit_children(array: &V::Array, visitor: &mut dyn ArrayChildVisitor);

    /// Count the number of children in the array.
    fn nchildren(array: &V::Array) -> usize {
        struct NChildren(usize);

        impl ArrayChildVisitor for NChildren {
            fn visit_child(&mut self, _name: &str, _array: &dyn Array) {
                self.0 += 1;
            }
        }

        let mut visitor = NChildren(0);
        <V::VisitorVTable as VisitorVTable<V>>::visit_children(array, &mut visitor);
        visitor.0
    }

    /// Replace the children of this array with the given arrays.
    ///
    /// ## Pre-conditions
    ///
    /// - The number of given children matches the current number of children of the array.
    // TODO(ngates): pass a Vec<ArrayRef> so the implementation can take ownership
    fn with_children(array: &V::Array, children: &[ArrayRef]) -> VortexResult<V::Array>;
}
