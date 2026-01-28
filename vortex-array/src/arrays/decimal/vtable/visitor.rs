// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::arrays::DecimalArray;
use crate::arrays::DecimalVTable;
use crate::vtable::ValidityHelper;
use crate::vtable::VisitorVTable;

impl VisitorVTable<DecimalVTable> for DecimalVTable {
    fn visit_buffers(array: &DecimalArray, visitor: &mut dyn ArrayBufferVisitor) {
        visitor
            .visit_buffer_handle(&array.values)
            .vortex_expect("Failed to visit buffer");
    }

    fn visit_children(array: &DecimalArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(array.validity(), array.len())
    }
}
