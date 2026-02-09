// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::ArrayRef;
use crate::arrays::BoolArray;
use crate::arrays::BoolVTable;
use crate::vtable::VisitorVTable;
use crate::vtable::validity_nchildren;
use crate::vtable::validity_to_child;

impl VisitorVTable<BoolVTable> for BoolVTable {
    fn visit_buffers(array: &BoolArray, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer_handle("bits", &array.bits);
    }

    fn nbuffers(_array: &BoolArray) -> usize {
        1
    }

    fn visit_children(array: &BoolArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(&array.validity, array.len());
    }

    fn nchildren(array: &BoolArray) -> usize {
        validity_nchildren(&array.validity)
    }

    fn nth_child(array: &BoolArray, idx: usize) -> Option<ArrayRef> {
        match idx {
            0 => validity_to_child(&array.validity, array.len()),
            _ => None,
        }
    }
}
