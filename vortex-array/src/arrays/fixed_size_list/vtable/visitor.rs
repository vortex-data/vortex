// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::arrays::FixedSizeListArray;
use crate::arrays::FixedSizeListVTable;
use crate::vtable::ValidityHelper;
use crate::vtable::VisitorVTable;

impl VisitorVTable<FixedSizeListVTable> for FixedSizeListVTable {
    fn visit_buffers(_array: &FixedSizeListArray, _visitor: &mut dyn ArrayBufferVisitor) {
        // `FixedSizeListArray` has no byte buffers.
    }

    // We define the children for [`FixedSizeListArray`] as the `elements` array and the `validity`.
    fn visit_children(array: &FixedSizeListArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("elements", array.elements());
        visitor.visit_validity(array.validity(), array.len());
    }
}
