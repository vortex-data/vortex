//! Utilities to traverse array trees using the visitor pattern.

use vortex_buffer::ByteBuffer;
use vortex_error::{VortexExpect, VortexResult};

use crate::patches::Patches;
use crate::validity::Validity;
use crate::{Array, ArrayRef};

pub trait ArrayVisitor {
    /// Visit a child of this array.
    fn visit_child(&mut self, _name: &str, _array: &dyn Array) -> VortexResult<()> {
        Ok(())
    }

    /// Utility for visiting Array validity.
    fn visit_validity(&mut self, validity: &Validity) -> VortexResult<()> {
        if let Some(v) = validity.as_array() {
            self.visit_child("validity", v)
        } else {
            Ok(())
        }
    }

    /// Utility for visiting Array patches.
    fn visit_patches(&mut self, patches: &Patches) -> VortexResult<()> {
        self.visit_child("patch_indices", patches.indices())?;
        self.visit_child("patch_values", patches.values())
    }

    fn visit_buffer(&mut self, _buffer: &ByteBuffer) -> VortexResult<()> {
        Ok(())
    }
}

/// Extension trait for [`Array`] that provides utility methods for inspecting array structures.
///
/// These functions aren't necessarily the fastest, but they all leverage the [`ArrayVisitor`] and
/// therefore are hard to implement incorrectly.
pub trait ArrayVisitorExt: Array {
    /// Returns the number of children of the array.
    fn nchildren(&self) -> usize {
        struct ChildrenCollector {
            nchildren: usize,
        }

        impl ArrayVisitor for ChildrenCollector {
            fn visit_child(&mut self, _name: &str, _array: &dyn Array) -> VortexResult<()> {
                self.nchildren += 1;
                Ok(())
            }
        }

        let mut collector = ChildrenCollector { nchildren: 0 };
        self.accept(&mut collector).vortex_expect("infallible");
        collector.nchildren
    }

    /// Returns the children of the array.
    fn children(&self) -> Vec<ArrayRef> {
        struct ChildrenCollector {
            children: Vec<ArrayRef>,
        }

        impl ArrayVisitor for ChildrenCollector {
            fn visit_child(&mut self, _name: &str, array: &dyn Array) -> VortexResult<()> {
                self.children.push(array.to_array());
                Ok(())
            }
        }

        let mut collector = ChildrenCollector { children: vec![] };
        self.accept(&mut collector).vortex_expect("infallible");
        collector.children
    }

    /// Returns the names of the children of the array.
    fn child_names(&self) -> Vec<Arc<str>> {
        struct NameCollector {
            names: Vec<Arc<str>>,
        }

        impl ArrayVisitor for NameCollector {
            fn visit_child(&mut self, name: &str, _array: &dyn Array) -> VortexResult<()> {
                self.names.push(name.into());
                Ok(())
            }
        }

        let mut collector = NameCollector { names: vec![] };
        self.accept(&mut collector).vortex_expect("infallible");
        collector.names
    }

    /// Returns the number of buffers of the array.
    fn nbuffers(&self) -> usize {
        struct BufferCollector {
            nchildren: usize,
        }

        impl ArrayVisitor for BufferCollector {
            fn visit_child(&mut self, _name: &str, _array: &dyn Array) -> VortexResult<()> {
                self.nchildren += 1;
                Ok(())
            }
        }

        let mut collector = BufferCollector { nchildren: 0 };
        self.accept(&mut collector).vortex_expect("infallible");
        collector.nchildren
    }

    /// Returns a vector of [`ByteBuffer`] for the array.
    fn byte_buffers(&self) -> Vec<ByteBuffer> {
        struct BufferCollector {
            buffers: Vec<ByteBuffer>,
        }

        impl ArrayVisitor for BufferCollector {
            fn visit_buffer(&mut self, buffer: &ByteBuffer) -> VortexResult<()> {
                self.buffers.push(buffer.clone());
                Ok(())
            }
        }

        let mut collector = BufferCollector { buffers: vec![] };
        self.accept(&mut collector).vortex_expect("infallible");
        collector.buffers
    }

    /// Depth-first traversal of the array and its children.
    fn depth_first_traversal(&self) -> impl Iterator<Item = ArrayRef> {
        /// A depth-first pre-order iterator over an Array.
        struct ArrayChildrenIterator {
            stack: Vec<ArrayRef>,
        }

        impl ArrayChildrenIterator {
            pub fn new(array: ArrayRef) -> Self {
                Self { stack: vec![array] }
            }
        }

        impl Iterator for ArrayChildrenIterator {
            type Item = ArrayRef;

            fn next(&mut self) -> Option<Self::Item> {
                let next = self.stack.pop()?;
                for child in next.children().into_iter().rev() {
                    self.stack.push(child);
                }
                Some(next)
            }
        }

        ArrayChildrenIterator {
            stack: vec![self.to_array()],
        }
    }
}

impl<A: Array + ?Sized> ArrayVisitorExt for A {}
