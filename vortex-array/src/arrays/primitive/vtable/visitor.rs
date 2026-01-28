// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::arrays::PrimitiveArray;
use crate::arrays::PrimitiveVTable;
use crate::vtable::ValidityHelper;
use crate::vtable::VisitorVTable;

impl VisitorVTable<PrimitiveVTable> for PrimitiveVTable {
    fn visit_buffers(array: &PrimitiveArray, visitor: &mut dyn ArrayBufferVisitor) {
        visitor
            .visit_buffer_handle("values", array.buffer_handle())
            .vortex_expect("Failed to visit buffer");
    }

    fn visit_children(array: &PrimitiveArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(array.validity(), array.len());
    }
}
