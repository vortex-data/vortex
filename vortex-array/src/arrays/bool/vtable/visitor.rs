// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::arrays::BoolArray;
use crate::arrays::BoolVTable;
use crate::vtable::VisitorVTable;

impl VisitorVTable<BoolVTable> for BoolVTable {
    fn visit_buffers(array: &BoolArray, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(array.bits.as_host())
    }

    fn visit_children(array: &BoolArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(&array.validity, array.len());
    }
}
