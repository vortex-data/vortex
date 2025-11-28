// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::arrays::scalar_fn::array::ScalarFnArray;
use crate::arrays::scalar_fn::vtable::ScalarFnVTable;
use crate::vtable::VisitorVTable;

impl VisitorVTable<ScalarFnVTable> for ScalarFnVTable {
    fn visit_buffers(_array: &ScalarFnArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &ScalarFnArray, visitor: &mut dyn ArrayChildVisitor) {
        for (idx, child) in array.children().iter().enumerate() {
            let name = array.scalar_fn.signature().arg_name(idx);
            visitor.visit_child(name.as_ref(), child.as_ref())
        }
    }
}
