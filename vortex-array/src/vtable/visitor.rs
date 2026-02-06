// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::ArrayChildVisitorUnnamed;
use crate::ArrayRef;
use crate::buffer::BufferHandle;
use crate::vtable::VTable;

pub trait VisitorVTable<V: VTable> {
    /// Visit the buffers of the array.
    fn visit_buffers(array: &V::Array, visitor: &mut dyn ArrayBufferVisitor);

    /// Count the number of buffers in the array.
    fn nbuffers(array: &V::Array) -> usize {
        struct NBuffers(usize);

        impl ArrayBufferVisitor for NBuffers {
            fn visit_buffer_handle(&mut self, _name: &str, _handle: &BufferHandle) {
                self.0 += 1;
            }
        }

        let mut visitor = NBuffers(0);
        <V::VisitorVTable as VisitorVTable<V>>::visit_buffers(array, &mut visitor);
        visitor.0
    }

    /// Return the names of the buffers in the array.
    fn buffer_names(array: &V::Array) -> Vec<String> {
        struct BufferNames(Vec<String>);

        impl ArrayBufferVisitor for BufferNames {
            fn visit_buffer_handle(&mut self, name: &str, _handle: &BufferHandle) {
                self.0.push(name.to_string());
            }
        }

        let mut visitor = BufferNames(Vec::new());
        <V::VisitorVTable as VisitorVTable<V>>::visit_buffers(array, &mut visitor);
        visitor.0
    }

    /// Visit the children of the array.
    fn visit_children(array: &V::Array, visitor: &mut dyn ArrayChildVisitor);

    /// Visit the children of the array without names.
    ///
    /// This is more efficient than [`Self::visit_children`] when you don't need the
    /// child names (e.g., for counting or accessing by index). The default
    /// implementation wraps the named visitor, but array types can override
    /// this to avoid allocating names.
    fn visit_children_unnamed(array: &V::Array, visitor: &mut dyn ArrayChildVisitorUnnamed) {
        struct UnnamedWrapper<'a>(&'a mut dyn ArrayChildVisitorUnnamed);

        impl ArrayChildVisitor for UnnamedWrapper<'_> {
            fn visit_child(&mut self, _name: &str, array: &ArrayRef) {
                self.0.visit_child(array);
            }
        }

        <V::VisitorVTable as VisitorVTable<V>>::visit_children(array, &mut UnnamedWrapper(visitor));
    }

    /// Count the number of children in the array.
    fn nchildren(array: &V::Array) -> usize {
        struct NChildren(usize);

        impl ArrayChildVisitorUnnamed for NChildren {
            fn visit_child(&mut self, _array: &ArrayRef) {
                self.0 += 1;
            }
        }

        let mut visitor = NChildren(0);
        <V::VisitorVTable as VisitorVTable<V>>::visit_children_unnamed(array, &mut visitor);
        visitor.0
    }

    /// Get the nth child of the array without allocating a Vec.
    ///
    /// Returns `None` if the index is out of bounds.
    fn nth_child(array: &V::Array, idx: usize) -> Option<ArrayRef> {
        struct NthChildVisitor {
            target_idx: usize,
            current_idx: usize,
            result: Option<ArrayRef>,
        }

        impl ArrayChildVisitorUnnamed for NthChildVisitor {
            fn visit_child(&mut self, array: &ArrayRef) {
                if self.current_idx == self.target_idx && self.result.is_none() {
                    self.result = Some(array.clone());
                }
                self.current_idx += 1;
            }
        }

        let mut visitor = NthChildVisitor {
            target_idx: idx,
            current_idx: 0,
            result: None,
        };
        <V::VisitorVTable as VisitorVTable<V>>::visit_children_unnamed(array, &mut visitor);
        visitor.result
    }
}
