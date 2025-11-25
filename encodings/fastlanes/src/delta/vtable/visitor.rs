// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayBufferVisitor;
use vortex_array::ArrayChildVisitor;
use vortex_array::vtable::VisitorVTable;

use super::DeltaVTable;
use crate::DeltaArray;

impl VisitorVTable<DeltaVTable> for DeltaVTable {
    fn visit_buffers(_array: &DeltaArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &DeltaArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("bases", array.bases());
        visitor.visit_child("deltas", array.deltas());
    }
}
