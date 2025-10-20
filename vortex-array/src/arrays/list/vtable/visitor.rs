// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::PType;

use super::serde::ListMetadata;
use crate::arrays::{ListArray, ListVTable};
use crate::vtable::{VTable, ValidityHelper, VisitorVTable};
use crate::{ArrayBufferVisitor, ArrayChildVisitor, ProstMetadata};

impl VisitorVTable<ListVTable> for ListVTable {
    fn metadata(array: &ListArray) -> <ListVTable as VTable>::Metadata {
        ProstMetadata(ListMetadata {
            elements_len: array.elements().len() as u64,
            offset_ptype: PType::try_from(array.offsets().dtype())
                .expect("Invalid PType for offsets") as i32,
        })
    }

    fn visit_buffers(_array: &ListArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &ListArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("elements", array.elements());
        visitor.visit_child("offsets", array.offsets());
        visitor.visit_validity(array.validity(), array.len());
    }
}
