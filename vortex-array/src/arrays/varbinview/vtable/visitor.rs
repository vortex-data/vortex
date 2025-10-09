// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use super::VarBinViewVTable;
use crate::arrays::VarBinViewArray;
use crate::vtable::{
    ValidityHelper,
    VisitorVTable,
};
use crate::{
    ArrayBufferVisitor,
    ArrayChildVisitor,
};

impl VisitorVTable<VarBinViewVTable> for VarBinViewVTable {
    fn visit_buffers(array: &VarBinViewArray, visitor: &mut dyn ArrayBufferVisitor) {
        for buffer in array.buffers().as_ref() {
            visitor.visit_buffer(buffer);
        }
        visitor.visit_buffer(&array.views().clone().into_byte_buffer());
    }

    fn visit_children(array: &VarBinViewArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(array.validity(), array.len())
    }
}
