// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::ArrayRef;
use crate::arrays::ListViewArray;
use crate::arrays::ListViewVTable;
use crate::vtable::ValidityHelper;
use crate::vtable::VisitorVTable;
use crate::vtable::validity_nchildren;
use crate::vtable::validity_to_child;

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

    fn nchildren(array: &ListViewArray) -> usize {
        3 + validity_nchildren(array.validity())
    }

    fn nth_child(array: &ListViewArray, idx: usize) -> Option<ArrayRef> {
        match idx {
            0 => Some(array.elements().clone()),
            1 => Some(array.offsets().clone()),
            2 => Some(array.sizes().clone()),
            3 => validity_to_child(array.validity(), array.len()),
            _ => None,
        }
    }
}
