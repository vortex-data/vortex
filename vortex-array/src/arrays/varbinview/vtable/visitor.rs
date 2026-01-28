// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;

use super::VarBinViewVTable;
use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::arrays::VarBinViewArray;
use crate::buffer::BufferHandle;
use crate::vtable::ValidityHelper;
use crate::vtable::VisitorVTable;

impl VisitorVTable<VarBinViewVTable> for VarBinViewVTable {
    fn visit_buffers(array: &VarBinViewArray, visitor: &mut dyn ArrayBufferVisitor) {
        for buffer in array.buffers().as_ref() {
            visitor
                .visit_buffer_handle(&BufferHandle::new_host(buffer.clone()))
                .vortex_expect("Failed to visit buffer");
        }
        visitor
            .visit_buffer_handle(&BufferHandle::new_host(
                array.views().clone().into_byte_buffer(),
            ))
            .vortex_expect("Failed to visit buffer");
    }

    fn visit_children(array: &VarBinViewArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(array.validity(), array.len())
    }
}
