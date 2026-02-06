// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::ArrayRef;
use crate::arrays::DecimalArray;
use crate::arrays::DecimalVTable;
use crate::vtable::ValidityHelper;
use crate::vtable::VisitorVTable;
use crate::vtable::validity_nchildren;
use crate::vtable::validity_to_child;

impl VisitorVTable<DecimalVTable> for DecimalVTable {
    fn visit_buffers(array: &DecimalArray, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer_handle("values", &array.values);
    }

    fn visit_children(array: &DecimalArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(array.validity(), array.len())
    }

    fn nchildren(array: &DecimalArray) -> usize {
        validity_nchildren(array.validity())
    }

    fn nth_child(array: &DecimalArray, idx: usize) -> Option<ArrayRef> {
        match idx {
            0 => validity_to_child(array.validity(), array.len()),
            _ => None,
        }
    }
}
