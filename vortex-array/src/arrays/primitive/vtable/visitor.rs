// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrays::PrimitiveArray;
use crate::arrays::PrimitiveVTable;
use crate::vtable::ValidityHelper;
use crate::vtable::VisitorVTable;

impl VisitorVTable<PrimitiveVTable> for PrimitiveVTable {
    fn visit_buffers(array: &PrimitiveArray, visitor: &mut dyn ArrayBufferVisitor) {
        let ctx = LEGACY_SESSION.create_execution_ctx();
        visitor.visit_buffer(array.buffer_handle(&ctx).bytes());
    }

    fn visit_children(array: &PrimitiveArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(array.validity(), array.len());
    }
}
