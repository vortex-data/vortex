// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use super::VarBinViewVTable;
use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::ArrayRef;
use crate::arrays::VarBinViewArray;
use crate::vtable::ValidityHelper;
use crate::vtable::VisitorVTable;
use crate::vtable::validity_nchildren;
use crate::vtable::validity_to_child;

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

    fn nchildren(array: &VarBinViewArray) -> usize {
        validity_nchildren(array.validity())
    }

    fn nth_child(array: &VarBinViewArray, idx: usize) -> Option<ArrayRef> {
        match idx {
            0 => validity_to_child(array.validity(), array.len()),
            _ => None,
        }
    }
}
