// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::{Alignment, ByteBuffer};

use crate::arrays::{BoolArray, BoolVTable};
use crate::vtable::VisitorVTable;
use crate::{ArrayBufferVisitor, ArrayChildVisitor};

impl VisitorVTable<BoolVTable> for BoolVTable {
    fn visit_buffers(array: &BoolArray, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(&ByteBuffer::from_arrow_buffer(
            array.boolean_buffer().clone().into_inner(),
            Alignment::none(),
        ))
    }

    fn visit_children(array: &BoolArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(&array.validity, array.len());
    }
}
