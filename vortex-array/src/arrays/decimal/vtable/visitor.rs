// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::arrays::DecimalArray;
use crate::arrays::DecimalVTable;
use crate::vtable::ValidityHelper;
use crate::vtable::VisitorVTable;

impl VisitorVTable<DecimalVTable> for DecimalVTable {
    fn visit_buffers(array: &DecimalArray, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(&array.values);
    }

    fn visit_children(array: &DecimalArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(array.validity(), array.len())
    }
}
