// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::{DecimalArray, DecimalVTable};
use crate::vtable::{ValidityHelper, VisitorVTable};
use crate::{ArrayBufferVisitor, ArrayChildVisitor};

impl VisitorVTable<DecimalVTable> for DecimalVTable {
    fn visit_buffers(array: &DecimalArray, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(&array.values);
    }

    fn visit_children<'a>(array: &'a DecimalArray, visitor: &mut dyn ArrayChildVisitor<'a>) {
        visitor.visit_validity(array.validity(), array.len())
    }
}
