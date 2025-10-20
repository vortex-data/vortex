// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::ByteBufferMut;

use crate::arrays::{ConstantArray, ConstantVTable};
use crate::vtable::VisitorVTable;
use crate::{ArrayBufferVisitor, ArrayChildVisitor, EmptyMetadata};

impl VisitorVTable<ConstantVTable> for ConstantVTable {
    fn metadata(_array: &ConstantArray) -> EmptyMetadata {
        EmptyMetadata
    }

    fn visit_buffers(array: &ConstantArray, visitor: &mut dyn ArrayBufferVisitor) {
        let buffer = array
            .scalar
            .value()
            .to_protobytes::<ByteBufferMut>()
            .freeze();
        visitor.visit_buffer(&buffer);
    }

    fn visit_children(_array: &ConstantArray, _visitor: &mut dyn ArrayChildVisitor) {}
}
