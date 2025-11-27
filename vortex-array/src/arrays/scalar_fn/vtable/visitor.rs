// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::scalar_fn::array::ScalarFnArray;
use crate::arrays::scalar_fn::vtable::ScalarFnVTable;
use crate::vtable::VisitorVTable;
use crate::{ArrayBufferVisitor, ArrayChildVisitor};

impl VisitorVTable<ScalarFnVTable> for ScalarFnVTable {
    fn visit_buffers(_array: &ScalarFnArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &ScalarFnArray, visitor: &mut dyn ArrayChildVisitor) {
        for (idx, child) in array.children().iter().enumerate() {
            let name = array.scalar_fn.signature().name(idx);
            visitor.visit_child(name.as_deref().unwrap_or_else(|| "unnamed"), child.as_ref())
        }
    }
}
