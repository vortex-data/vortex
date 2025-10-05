// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::{PrimitiveArray, PrimitiveVTable};
use crate::vtable::{ValidityHelper, VisitorVTable};
use crate::{ArrayBufferVisitor, ArrayChildVisitor};

impl VisitorVTable<PrimitiveVTable> for PrimitiveVTable {
    fn visit_buffers(array: &PrimitiveArray, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(array.byte_buffer());
    }

    fn visit_children(array: &PrimitiveArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(array.validity(), array.len());
    }
}
