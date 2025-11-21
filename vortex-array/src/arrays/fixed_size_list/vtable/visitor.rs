// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::{FixedSizeListArray, FixedSizeListVTable};
use crate::vtable::{ValidityHelper, VisitorVTable};
use crate::{ArrayBufferVisitor, ArrayChildVisitor};

impl VisitorVTable<FixedSizeListVTable> for FixedSizeListVTable {
    fn visit_buffers(_array: &FixedSizeListArray, _visitor: &mut dyn ArrayBufferVisitor) {
        // `FixedSizeListArray` has no byte buffers.
    }

    // We define the children for [`FixedSizeListArray`] as the `elements` array and the `validity`.
    fn visit_children<'a>(array: &'a FixedSizeListArray, visitor: &mut dyn ArrayChildVisitor<'a>) {
        visitor.visit_child("elements", array.elements());
        visitor.visit_validity(array.validity(), array.len());
    }
}
