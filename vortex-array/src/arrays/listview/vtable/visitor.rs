// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use super::serde::ListViewMetadata;
use crate::arrays::{ListViewArray, ListViewVTable};
use crate::vtable::{VTable, ValidityHelper, VisitorVTable};
use crate::{ArrayBufferVisitor, ArrayChildVisitor, ProstMetadata};

impl VisitorVTable<ListViewVTable> for ListViewVTable {
    fn metadata(array: &ListViewArray) -> <ListViewVTable as VTable>::Metadata {
        ProstMetadata(ListViewMetadata {
            elements_len: array.elements().len() as u64,
            offset_ptype: array.offsets().dtype().as_ptype() as i32,
            size_ptype: array.sizes().dtype().as_ptype() as i32,
        })
    }

    fn visit_buffers(_array: &ListViewArray, _visitor: &mut dyn ArrayBufferVisitor) {
        // ListView has no byte buffers.
    }

    fn visit_children(array: &ListViewArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("elements", array.elements());
        visitor.visit_child("offsets", array.offsets());
        visitor.visit_child("sizes", array.sizes());
        visitor.visit_validity(array.validity(), array.len());
    }
}
