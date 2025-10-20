// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::extension::{ExtensionArray, ExtensionVTable};
use crate::vtable::VisitorVTable;
use crate::{ArrayBufferVisitor, ArrayChildVisitor, EmptyMetadata};

impl VisitorVTable<ExtensionVTable> for ExtensionVTable {
    fn metadata(_array: &ExtensionArray) -> EmptyMetadata {
        EmptyMetadata
    }

    fn visit_buffers(_array: &ExtensionArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &ExtensionArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("storage", array.storage.as_ref());
    }
}
