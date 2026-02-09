// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::ArrayRef;
use crate::arrays::ListArray;
use crate::arrays::ListVTable;
use crate::vtable::ValidityHelper;
use crate::vtable::VisitorVTable;
use crate::vtable::validity_nchildren;
use crate::vtable::validity_to_child;

impl VisitorVTable<ListVTable> for ListVTable {
    fn visit_buffers(_array: &ListArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn nbuffers(_array: &ListArray) -> usize {
        0
    }

    fn visit_children(array: &ListArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("elements", array.elements());
        visitor.visit_child("offsets", array.offsets());
        visitor.visit_validity(array.validity(), array.len());
    }

    fn nchildren(array: &ListArray) -> usize {
        2 + validity_nchildren(array.validity())
    }

    fn nth_child(array: &ListArray, idx: usize) -> Option<ArrayRef> {
        match idx {
            0 => Some(array.elements().clone()),
            1 => Some(array.offsets().clone()),
            2 => validity_to_child(array.validity(), array.len()),
            _ => None,
        }
    }
}
