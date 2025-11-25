// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::arrays::expr::ExprArray;
use crate::arrays::expr::ExprVTable;
use crate::vtable::VisitorVTable;

impl VisitorVTable<ExprVTable> for ExprVTable {
    fn visit_buffers(_array: &ExprArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &ExprArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("child", &array.child);
    }
}
