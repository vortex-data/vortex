// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayBufferVisitor;
use vortex_array::ArrayChildVisitor;
use vortex_array::vtable::ValidityHelper;
use vortex_array::vtable::VisitorVTable;
use vortex_array::vtable::validity_nchildren;

use crate::BitPackedArray;
use crate::BitPackedVTable;

impl VisitorVTable<BitPackedVTable> for BitPackedVTable {
    fn visit_buffers(array: &BitPackedArray, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer_handle("packed", array.packed());
    }

    fn nbuffers(_array: &BitPackedArray) -> usize {
        1
    }

    fn visit_children(array: &BitPackedArray, visitor: &mut dyn ArrayChildVisitor) {
        if let Some(patches) = array.patches() {
            visitor.visit_patches(patches);
        }
        visitor.visit_validity(array.validity(), array.len());
    }

    fn nchildren(array: &BitPackedArray) -> usize {
        // optional patches (indices + values + optional chunk_offsets) + optional validity
        array
            .patches()
            .map_or(0, |p| 2 + p.chunk_offsets().is_some() as usize)
            + validity_nchildren(array.validity())
    }
}
