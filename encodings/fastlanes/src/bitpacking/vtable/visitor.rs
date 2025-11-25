// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayBufferVisitor;
use vortex_array::ArrayChildVisitor;
use vortex_array::vtable::ValidityHelper;
use vortex_array::vtable::VisitorVTable;

use crate::BitPackedArray;
use crate::BitPackedVTable;

impl VisitorVTable<BitPackedVTable> for BitPackedVTable {
    fn visit_buffers(array: &BitPackedArray, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(array.packed());
    }

    fn visit_children(array: &BitPackedArray, visitor: &mut dyn ArrayChildVisitor) {
        if let Some(patches) = array.patches() {
            visitor.visit_patches(patches);
        }
        visitor.visit_validity(array.validity(), array.len());
    }
}
