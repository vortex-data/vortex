use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::arrays::ConstantArray;
use crate::patches::Patches;
use crate::validity::Validity;
use crate::{
    Array, ArrayImpl, ArrayRef, ArrayValidityImpl, DeserializeMetadata, EmptyMetadata, Encoding,
    SerializeMetadata,
};

pub trait ArrayVisitor {
    /// Returns the children of the array.
    fn children(&self) -> Vec<ArrayRef>;

    /// Returns the number of children of the array.
    fn nchildren(&self) -> usize;

    /// Returns the names of the children of the array.
    fn children_names(&self) -> Vec<String>;

    /// Returns the buffers of the array.
    fn buffers(&self) -> Vec<ByteBuffer>;

    /// Returns the number of buffers of the array.
    fn nbuffers(&self) -> usize;

    /// Returns the serialized metadata of the array.
    fn metadata(&self) -> Option<Vec<u8>>;

    /// Formats a human-readable metadata description.
    fn metadata_fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result;
}

impl ArrayVisitor for Arc<dyn Array> {
    fn children(&self) -> Vec<ArrayRef> {
        self.as_ref().children()
    }

    fn nchildren(&self) -> usize {
        self.as_ref().nchildren()
    }

    fn children_names(&self) -> Vec<String> {
        self.as_ref().children_names()
    }

    fn buffers(&self) -> Vec<ByteBuffer> {
        self.as_ref().buffers()
    }

    fn nbuffers(&self) -> usize {
        self.as_ref().nbuffers()
    }

    fn metadata(&self) -> Option<Vec<u8>> {
        self.as_ref().metadata()
    }

    fn metadata_fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.as_ref().metadata_fmt(f)
    }
}

pub trait ArrayVisitorExt: Array {
    /// Count the number of buffers encoded by self and all child arrays.
    fn nbuffers_recursive(&self) -> usize {
        self.children()
            .iter()
            .map(ArrayVisitorExt::nbuffers_recursive)
            .sum::<usize>()
            + self.nbuffers()
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

// TODO(ngates): rename to ArraySerdeImpl?
pub trait ArrayVisitorImpl<
    Metadata: SerializeMetadata + DeserializeMetadata + Debug = EmptyMetadata,
>
{
    fn _buffers(&self, visitor: &mut dyn ArrayBufferVisitor) {}

    fn _nbuffers(&self) -> usize {
        struct NBuffers(usize);

        impl ArrayBufferVisitor for NBuffers {
            fn visit_buffer(&mut self, buffer: &ByteBuffer) {
                self.0 += 1;
            }
        }

        let mut visitor = NBuffers(0);
        self._buffers(&mut visitor);
        visitor.0
    }

    fn _children(&self, visitor: &mut dyn ArrayChildVisitor) {}

    fn _nchildren(&self) -> usize {
        struct NChildren(usize);

        impl ArrayChildVisitor for NChildren {
            fn visit_child(&mut self, _name: &str, _array: &dyn Array) {
                self.0 += 1;
            }
        }

        let mut visitor = NChildren(0);
        self._children(&mut visitor);
        visitor.0
    }

    fn _metadata(&self) -> Metadata;
}

pub trait ArrayBufferVisitor {
    fn visit_buffer(&mut self, buffer: &ByteBuffer);
}

pub trait ArrayChildVisitor {
    /// Visit a child of this array.
    fn visit_child(&mut self, _name: &str, _array: &dyn Array);

    /// Utility for visiting Array validity.
    fn visit_validity(&mut self, validity: &Validity, len: usize) {
        if let Some(vlen) = validity.maybe_len() {
            assert_eq!(vlen, len, "Validity length mismatch");
        }

        match validity {
            Validity::NonNullable | Validity::AllValid => {}
            Validity::AllInvalid => {
                // To avoid storing metadata about validity, we store all invalid as a
                // constant array of false values.
                // This gives:
                //  * is_nullable & has_validity => Validity::Array (or Validity::AllInvalid)
                //  * is_nullable & !has_validity => Validity::AllValid
                //  * !is_nullable => Validity::NonNullable
                self.visit_child("validity", &ConstantArray::new(false, len))
            }
            Validity::Array(array) => {
                self.visit_child("validity", array);
            }
        }
    }

    /// Utility for visiting Array patches.
    fn visit_patches(&mut self, patches: &Patches) {
        self.visit_child("patch_indices", patches.indices());
        self.visit_child("patch_values", patches.values());
    }
}

impl<A: ArrayImpl> ArrayVisitor for A {
    fn children(&self) -> Vec<ArrayRef> {
        struct ChildrenCollector {
            children: Vec<ArrayRef>,
        }

        impl ArrayChildVisitor for ChildrenCollector {
            fn visit_child(&mut self, _name: &str, array: &dyn Array) {
                self.children.push(array.to_array());
            }
        }

        let mut collector = ChildrenCollector {
            children: Vec::new(),
        };
        ArrayVisitorImpl::_children(self, &mut collector);
        collector.children
    }

    fn nchildren(&self) -> usize {
        ArrayVisitorImpl::_nchildren(self)
    }

    fn children_names(&self) -> Vec<String> {
        struct ChildNameCollector {
            names: Vec<String>,
        }

        impl ArrayChildVisitor for ChildNameCollector {
            fn visit_child(&mut self, name: &str, _array: &dyn Array) {
                self.names.push(name.to_string());
            }
        }

        let mut collector = ChildNameCollector { names: Vec::new() };
        ArrayVisitorImpl::_children(self, &mut collector);
        collector.names
    }

    fn buffers(&self) -> Vec<ByteBuffer> {
        struct BufferCollector {
            buffers: Vec<ByteBuffer>,
        }

        impl ArrayBufferVisitor for BufferCollector {
            fn visit_buffer(&mut self, buffer: &ByteBuffer) {
                self.buffers.push(buffer.clone());
            }
        }

        let mut collector = BufferCollector {
            buffers: Vec::new(),
        };
        ArrayVisitorImpl::_buffers(self, &mut collector);
        collector.buffers
    }

    fn nbuffers(&self) -> usize {
        ArrayVisitorImpl::_nbuffers(self)
    }

    fn metadata(&self) -> Option<Vec<u8>> {
        SerializeMetadata::serialize(&ArrayVisitorImpl::_metadata(self))
    }

    fn metadata_fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&ArrayVisitorImpl::_metadata(self), f)
    }
}
