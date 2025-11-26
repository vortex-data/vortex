// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::arrays::extension::ExtensionArray;
use crate::arrays::extension::ExtensionVTable;
use crate::vtable::VisitorVTable;

impl VisitorVTable<ExtensionVTable> for ExtensionVTable {
    fn visit_buffers(_array: &ExtensionArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &ExtensionArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("storage", array.storage.as_ref());
    }
}
