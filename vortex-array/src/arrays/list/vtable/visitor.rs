// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::arrays::ListArray;
use crate::arrays::ListVTable;
use crate::vtable::ValidityHelper;
use crate::vtable::VisitorVTable;

impl VisitorVTable<ListVTable> for ListVTable {
    fn visit_buffers(_array: &ListArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &ListArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("elements", array.elements());
        visitor.visit_child("offsets", array.offsets());
        visitor.visit_validity(array.validity(), array.len());
    }
}
