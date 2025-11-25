// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::arrays::ListViewArray;
use crate::arrays::ListViewVTable;
use crate::vtable::ValidityHelper;
use crate::vtable::VisitorVTable;

impl VisitorVTable<ListViewVTable> for ListViewVTable {
    fn visit_buffers(_array: &ListViewArray, _visitor: &mut dyn ArrayBufferVisitor) {
        // ListView has no byte buffers.
    }

    fn visit_children(array: &ListViewArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("elements", array.elements());
        visitor.visit_child("offsets", array.offsets());
        visitor.visit_child("sizes", array.sizes());
        visitor.visit_validity(array.validity(), array.len());
    }
}
