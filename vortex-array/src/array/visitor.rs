// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;
use std::sync::Arc;

use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::DynArray;
use crate::buffer::BufferHandle;

pub trait ArrayVisitor {
    /// Returns the children of the array.
    fn children(&self) -> Vec<ArrayRef>;

    /// Returns the number of children of the array.
    fn nchildren(&self) -> usize;

    /// Returns the nth child of the array without allocating a Vec.
    ///
    /// Returns `None` if the index is out of bounds.
    fn nth_child(&self, idx: usize) -> Option<ArrayRef>;

    /// Returns the names of the children of the array.
    fn children_names(&self) -> Vec<String>;

    /// Returns the slots of the array as a slice.
    fn slots(&self) -> &[Option<ArrayRef>];

    /// Returns the array's children with their names.
    fn named_children(&self) -> Vec<(String, ArrayRef)>;

    /// Returns the buffers of the array.
    fn buffers(&self) -> Vec<ByteBuffer>;

    /// Returns the buffer handles of the array.
    fn buffer_handles(&self) -> Vec<BufferHandle>;

    /// Returns the names of the buffers of the array.
    fn buffer_names(&self) -> Vec<String>;

    /// Returns the array's buffers with their names.
    fn named_buffers(&self) -> Vec<(String, BufferHandle)>;

    /// Returns the number of buffers of the array.
    fn nbuffers(&self) -> usize;

    /// Returns the serialized metadata of the array, or `None` if the array does not
    /// support serialization.
    fn metadata(&self) -> VortexResult<Option<Vec<u8>>>;

    /// Formats a human-readable metadata description.
    fn metadata_fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result;

    /// Checks if all buffers in the array tree are host-resident.
    ///
    /// This will fail if any buffers of self or child arrays are GPU-resident.
    fn is_host(&self) -> bool;
}

impl ArrayVisitor for Arc<dyn DynArray> {
    fn children(&self) -> Vec<ArrayRef> {
        self.as_ref().children()
    }

    fn nchildren(&self) -> usize {
        self.as_ref().nchildren()
    }

    fn nth_child(&self, idx: usize) -> Option<ArrayRef> {
        self.as_ref().nth_child(idx)
    }

    fn children_names(&self) -> Vec<String> {
        self.as_ref().children_names()
    }

    fn slots(&self) -> &[Option<ArrayRef>] {
        self.as_ref().slots()
    }

    fn named_children(&self) -> Vec<(String, ArrayRef)> {
        self.as_ref().named_children()
    }

    fn buffers(&self) -> Vec<ByteBuffer> {
        self.as_ref().buffers()
    }

    fn buffer_handles(&self) -> Vec<BufferHandle> {
        self.as_ref().buffer_handles()
    }

    fn buffer_names(&self) -> Vec<String> {
        self.as_ref().buffer_names()
    }

    fn named_buffers(&self) -> Vec<(String, BufferHandle)> {
        self.as_ref().named_buffers()
    }

    fn nbuffers(&self) -> usize {
        self.as_ref().nbuffers()
    }

    fn metadata(&self) -> VortexResult<Option<Vec<u8>>> {
        self.as_ref().metadata()
    }

    fn metadata_fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.as_ref().metadata_fmt(f)
    }

    fn is_host(&self) -> bool {
        self.as_ref().is_host()
    }
}

pub trait ArrayVisitorExt: DynArray {
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

impl<A: DynArray + ?Sized> ArrayVisitorExt for A {}
