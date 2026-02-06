// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use super::DictVTable;
use crate::ArrayBufferVisitor;
use crate::ArrayChildVisitor;
use crate::ArrayRef;
use crate::arrays::dict::DictArray;
use crate::vtable::VisitorVTable;

impl VisitorVTable<DictVTable> for DictVTable {
    fn visit_buffers(_array: &DictArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &DictArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("codes", array.codes());
        visitor.visit_child("values", array.values());
    }

    fn nchildren(_array: &DictArray) -> usize {
        2
    }

    fn nth_child(array: &DictArray, idx: usize) -> Option<ArrayRef> {
        match idx {
            0 => Some(array.codes().clone()),
            1 => Some(array.values().clone()),
            _ => None,
        }
    }
}
