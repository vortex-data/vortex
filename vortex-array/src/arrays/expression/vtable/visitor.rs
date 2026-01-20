// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::arrays::ExpressionArray;
use crate::arrays::ExpressionVTable;
use crate::vtable::VisitorVTable;

impl VisitorVTable<ExpressionVTable> for ExpressionVTable {
    fn visit_buffers(_array: &ExpressionArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &ExpressionArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("scope", &array.input);
    }
}
