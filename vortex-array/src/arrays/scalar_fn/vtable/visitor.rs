// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::ArrayChildVisitorUnnamed;
use crate::ArrayRef;
use crate::arrays::scalar_fn::array::ScalarFnArray;
use crate::arrays::scalar_fn::vtable::ScalarFnVTable;
use crate::vtable::VisitorVTable;

impl VisitorVTable<ScalarFnVTable> for ScalarFnVTable {
    fn visit_buffers(_array: &ScalarFnArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &ScalarFnArray, visitor: &mut dyn ArrayChildVisitor) {
        for (idx, child) in array.children.iter().enumerate() {
            let name = array.scalar_fn.signature().child_name(idx);
            visitor.visit_child(name.as_ref(), child)
        }
    }

    fn visit_children_unnamed(array: &ScalarFnArray, visitor: &mut dyn ArrayChildVisitorUnnamed) {
        for child in array.children.iter() {
            visitor.visit_child(child);
        }
    }

    fn nchildren(array: &ScalarFnArray) -> usize {
        array.children.len()
    }

    fn nth_child(array: &ScalarFnArray, idx: usize) -> Option<ArrayRef> {
        array.children.get(idx).cloned()
    }
}
