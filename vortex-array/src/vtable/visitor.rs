// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::ByteBuffer;

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::ArrayRef;
use crate::vtable::VTable;

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
            fn visit_child(&mut self, _name: &str, _array: &ArrayRef) {
                self.0 += 1;
            }
        }

        let mut visitor = NChildren(0);
        <V::VisitorVTable as VisitorVTable<V>>::visit_children(array, &mut visitor);
        visitor.0
    }
}
