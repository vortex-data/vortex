// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::arrays::VarBinArray;
use crate::arrays::VarBinVTable;
use crate::vtable::ValidityHelper;
use crate::vtable::VisitorVTable;

impl VisitorVTable<VarBinVTable> for VarBinVTable {
    fn visit_buffers(array: &VarBinArray, visitor: &mut dyn ArrayBufferVisitor) {
        // TODO(ngates): sliced bytes?
        visitor
            .visit_buffer_handle(array.bytes_handle())
            .vortex_expect("Failed to visit buffer");
    }

    fn visit_children(array: &VarBinArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("offsets", array.offsets());
        visitor.visit_validity(array.validity(), array.len());
    }
}
