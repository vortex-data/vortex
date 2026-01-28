// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::arrays::BoolArray;
use crate::arrays::BoolVTable;
use crate::vtable::VisitorVTable;

impl VisitorVTable<BoolVTable> for BoolVTable {
    fn visit_buffers(array: &BoolArray, visitor: &mut dyn ArrayBufferVisitor) {
        visitor
            .visit_buffer_handle("bits", &array.bits)
            .vortex_expect("Failed to visit buffer");
    }

    fn visit_children(array: &BoolArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(&array.validity, array.len());
    }
}
