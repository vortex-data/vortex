// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::arrays::PrimitiveArray;
use crate::arrays::PrimitiveVTable;
use crate::vtable::ValidityHelper;
use crate::vtable::VisitorVTable;

impl VisitorVTable<PrimitiveVTable> for PrimitiveVTable {
    fn visit_buffers(array: &PrimitiveArray, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(&array.buffer_handle().to_host_sync());
    }

    fn visit_children(array: &PrimitiveArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(array.validity(), array.len());
    }
}
