// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::ArrayChildVisitorUnnamed;
use crate::arrays::ChunkedArray;
use crate::arrays::ChunkedVTable;
use crate::vtable::VisitorVTable;

impl VisitorVTable<ChunkedVTable> for ChunkedVTable {
    fn visit_buffers(_array: &ChunkedArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn nbuffers(_array: &ChunkedArray) -> usize {
        0
    }

    fn visit_children(array: &ChunkedArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("chunk_offsets", &array.chunk_offsets.to_array());

        for (idx, chunk) in array.chunks().iter().enumerate() {
            visitor.visit_child(format!("chunks[{idx}]").as_str(), chunk);
        }
    }

    fn visit_children_unnamed(array: &ChunkedArray, visitor: &mut dyn ArrayChildVisitorUnnamed) {
        visitor.visit_child(&array.chunk_offsets.to_array());

        for chunk in array.chunks().iter() {
            visitor.visit_child(chunk);
        }
    }

    fn nchildren(array: &ChunkedArray) -> usize {
        1 + array.chunks().len()
    }

    fn nth_child(array: &ChunkedArray, idx: usize) -> Option<crate::ArrayRef> {
        match idx {
            0 => Some(array.chunk_offsets.to_array()),
            n => array.chunks().get(n - 1).cloned(),
        }
    }
}
