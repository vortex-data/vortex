// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::ArrayRef;
use crate::arrays::extension::ExtensionArray;
use crate::arrays::extension::ExtensionVTable;
use crate::vtable::VisitorVTable;

impl VisitorVTable<ExtensionVTable> for ExtensionVTable {
    fn visit_buffers(_array: &ExtensionArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn nbuffers(_array: &ExtensionArray) -> usize {
        0
    }

    fn visit_children(array: &ExtensionArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("storage", &array.storage);
    }

    fn nchildren(_array: &ExtensionArray) -> usize {
        1
    }

    fn nth_child(array: &ExtensionArray, idx: usize) -> Option<ArrayRef> {
        match idx {
            0 => Some(array.storage.clone()),
            _ => None,
        }
    }
}
