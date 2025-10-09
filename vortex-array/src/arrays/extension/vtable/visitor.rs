// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::extension::{
    ExtensionArray,
    ExtensionVTable,
};
use crate::vtable::VisitorVTable;
use crate::{
    ArrayBufferVisitor,
    ArrayChildVisitor,
};

impl VisitorVTable<ExtensionVTable> for ExtensionVTable {
    fn visit_buffers(_array: &ExtensionArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &ExtensionArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("storage", array.storage.as_ref());
    }
}
