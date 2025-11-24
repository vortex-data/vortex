// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::vtable::VisitorVTable;
use vortex_array::{ArrayBufferVisitor, ArrayChildVisitor};

use super::RLEVTable;
use crate::RLEArray;

impl VisitorVTable<RLEVTable> for RLEVTable {
    fn visit_buffers(_array: &RLEArray, _visitor: &mut dyn ArrayBufferVisitor) {
        // RLE stores all data in child arrays, no direct buffers
    }

    fn visit_children(array: &RLEArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("values", array.values());
        visitor.visit_child("indices", array.indices());
        visitor.visit_child("values_idx_offsets", array.values_idx_offsets());
        // Don't call visit_validity since the nullability is stored in the indices array.
    }
}
