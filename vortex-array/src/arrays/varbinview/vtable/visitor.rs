// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use super::VarBinViewVTable;
use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::arrays::VarBinViewArray;
use crate::buffer::BufferHandle;
use crate::vtable::ValidityHelper;
use crate::vtable::VisitorVTable;

impl VisitorVTable<VarBinViewVTable> for VarBinViewVTable {
    fn visit_buffers(array: &VarBinViewArray, visitor: &mut dyn ArrayBufferVisitor) {
        for (i, buffer) in array.buffers().iter().enumerate() {
            visitor.visit_buffer_handle(
                &format!("buffer_{i}"),
                &BufferHandle::new_host(buffer.clone()),
            );
        }
        visitor.visit_buffer_handle(
            "views",
            &BufferHandle::new_host(array.views().clone().into_byte_buffer()),
        );
    }

    fn visit_children(array: &VarBinViewArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(array.validity(), array.len())
    }
}
