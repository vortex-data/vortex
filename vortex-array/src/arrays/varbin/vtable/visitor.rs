// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::arrays::VarBinArray;
use crate::arrays::VarBinVTable;
use crate::vtable::ValidityHelper;
use crate::vtable::VisitorVTable;

impl VisitorVTable<VarBinVTable> for VarBinVTable {
    fn visit_buffers(array: &VarBinArray, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(array.bytes()); // TODO(ngates): sliced bytes?
    }

    fn visit_children(array: &VarBinArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("offsets", array.offsets());
        visitor.visit_validity(array.validity(), array.len());
    }
}
