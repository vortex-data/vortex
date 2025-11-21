// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::vtable::{ValidityHelper, VisitorVTable};
use vortex_array::{ArrayBufferVisitor, ArrayChildVisitor};

use crate::{BitPackedArray, BitPackedVTable};

impl VisitorVTable<BitPackedVTable> for BitPackedVTable {
    fn visit_buffers(array: &BitPackedArray, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(array.packed());
    }

    fn visit_children<'a>(array: &'a BitPackedArray, visitor: &mut dyn ArrayChildVisitor<'a>) {
        if let Some(patches) = array.patches() {
            visitor.visit_patches(patches);
        }
        visitor.visit_validity(array.validity(), array.len());
    }
}
