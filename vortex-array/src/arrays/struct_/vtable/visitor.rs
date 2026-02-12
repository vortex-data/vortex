// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::ArrayChildVisitorUnnamed;
use crate::ArrayRef;
use crate::arrays::struct_::StructArray;
use crate::arrays::struct_::StructVTable;
use crate::vtable::ValidityHelper;
use crate::vtable::VisitorVTable;
use crate::vtable::validity_nchildren;
use crate::vtable::validity_to_child;

impl VisitorVTable<StructVTable> for StructVTable {
    fn visit_buffers(_array: &StructArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn nbuffers(_array: &StructArray) -> usize {
        0
    }

    fn visit_children(array: &StructArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(array.validity(), array.len());
        for (name, field) in array.names().iter().zip_eq(array.unmasked_fields().iter()) {
            visitor.visit_child(name.as_ref(), field);
        }
    }

    fn visit_children_unnamed(array: &StructArray, visitor: &mut dyn ArrayChildVisitorUnnamed) {
        visitor.visit_validity(array.validity(), array.len());
        for field in array.unmasked_fields().iter() {
            visitor.visit_child(field);
        }
    }

    fn nchildren(array: &StructArray) -> usize {
        validity_nchildren(array.validity()) + array.unmasked_fields().len()
    }

    fn nth_child(array: &StructArray, idx: usize) -> Option<ArrayRef> {
        let validity_children = validity_nchildren(array.validity());
        if idx < validity_children {
            validity_to_child(array.validity(), array.len())
        } else {
            array
                .unmasked_fields()
                .get(idx - validity_children)
                .cloned()
        }
    }
}
