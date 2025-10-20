// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;

use crate::arrays::struct_::{StructArray, StructVTable};
use crate::vtable::{ValidityHelper, VisitorVTable};
use crate::{ArrayBufferVisitor, ArrayChildVisitor};

impl VisitorVTable<StructVTable> for StructVTable {
    fn visit_buffers(_array: &StructArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &StructArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(array.validity(), array.len());
        for (name, field) in array.names().iter().zip_eq(array.columns().iter()) {
            visitor.visit_child(name.as_ref(), field);
        }
    }
}
