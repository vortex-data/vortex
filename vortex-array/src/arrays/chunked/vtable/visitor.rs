// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::arrays::ChunkedArray;
use crate::arrays::ChunkedVTable;
use crate::vtable::VisitorVTable;

impl VisitorVTable<ChunkedVTable> for ChunkedVTable {
    fn visit_buffers(_array: &ChunkedArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &ChunkedArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("chunk_offsets", array.chunk_offsets.as_ref());

        for (idx, chunk) in array.chunks().iter().enumerate() {
            visitor.visit_child(format!("chunks[{idx}]").as_str(), chunk);
        }
    }
}
