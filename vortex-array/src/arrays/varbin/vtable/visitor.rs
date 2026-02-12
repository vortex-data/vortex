// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::ArrayRef;
use crate::arrays::VarBinArray;
use crate::arrays::VarBinVTable;
use crate::vtable::ValidityHelper;
use crate::vtable::VisitorVTable;
use crate::vtable::validity_nchildren;
use crate::vtable::validity_to_child;

impl VisitorVTable<VarBinVTable> for VarBinVTable {
    fn visit_buffers(array: &VarBinArray, visitor: &mut dyn ArrayBufferVisitor) {
        // TODO(ngates): sliced bytes?
        visitor.visit_buffer_handle("bytes", array.bytes_handle());
    }

    fn nbuffers(_array: &VarBinArray) -> usize {
        1
    }

    fn visit_children(array: &VarBinArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("offsets", array.offsets());
        visitor.visit_validity(array.validity(), array.len());
    }

    fn nchildren(array: &VarBinArray) -> usize {
        1 + validity_nchildren(array.validity())
    }

    fn nth_child(array: &VarBinArray, idx: usize) -> Option<ArrayRef> {
        match idx {
            0 => Some(array.offsets().clone()),
            1 => validity_to_child(array.validity(), array.len()),
            _ => None,
        }
    }
}
