// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayBufferVisitor;
use vortex_array::ArrayChildVisitor;
use vortex_array::vtable::VisitorVTable;

use super::FoRVTable;
use crate::FoRArray;

impl VisitorVTable<FoRVTable> for FoRVTable {
    fn visit_buffers(_array: &FoRArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn nbuffers(_array: &FoRArray) -> usize {
        0
    }

    fn visit_children(array: &FoRArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("encoded", array.encoded())
    }

    fn nchildren(_array: &FoRArray) -> usize {
        1
    }
}
