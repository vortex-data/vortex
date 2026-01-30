// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use super::VarBinViewVTable;
use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::arrays::VarBinViewArray;
use crate::vtable::ValidityHelper;
use crate::vtable::VisitorVTable;

impl VisitorVTable<VarBinViewVTable> for VarBinViewVTable {
    fn visit_buffers(array: &VarBinViewArray, visitor: &mut dyn ArrayBufferVisitor) {
        for (i, buffer) in array.buffers().iter().enumerate() {
            visitor.visit_buffer_handle(&format!("buffer_{i}"), buffer);
        }
        visitor.visit_buffer_handle("views", array.views_handle());
    }

    fn visit_children(array: &VarBinViewArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(array.validity(), array.len())
    }
}
