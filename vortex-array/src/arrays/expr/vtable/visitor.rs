// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::expr::{ExprArray, ExprVTable};
use crate::vtable::VisitorVTable;
use crate::{ArrayBufferVisitor, ArrayChildVisitor};

impl VisitorVTable<ExprVTable> for ExprVTable {
    fn visit_buffers(_array: &ExprArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &ExprArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("child", array.child.as_ref());
    }
}
