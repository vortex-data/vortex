// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::ByteBufferMut;

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::ArrayRef;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::buffer::BufferHandle;
use crate::scalar::ScalarValue;
use crate::vtable::VisitorVTable;

impl VisitorVTable<ConstantVTable> for ConstantVTable {
    fn visit_buffers(array: &ConstantArray, visitor: &mut dyn ArrayBufferVisitor) {
        let buffer = ScalarValue::to_proto_bytes::<ByteBufferMut>(array.scalar.value()).freeze();
        visitor.visit_buffer_handle("scalar", &BufferHandle::new_host(buffer));
    }

    fn nbuffers(_array: &ConstantArray) -> usize {
        1
    }

    fn visit_children(_array: &ConstantArray, _visitor: &mut dyn ArrayChildVisitor) {}

    fn nchildren(_array: &ConstantArray) -> usize {
        0
    }

    fn nth_child(_array: &ConstantArray, _idx: usize) -> Option<ArrayRef> {
        None
    }
}
