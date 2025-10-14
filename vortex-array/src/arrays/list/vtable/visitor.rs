// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::{ListArray, ListVTable};
use crate::vtable::{ValidityHelper, VisitorVTable};
use crate::{ArrayBufferVisitor, ArrayChildVisitor};

impl VisitorVTable<ListVTable> for ListVTable {
    fn visit_buffers(_array: &ListArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &ListArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("elements", array.elements());
        visitor.visit_child("offsets", array.offsets());
        visitor.visit_validity(array.validity(), array.len());
    }
}
