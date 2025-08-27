// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::{FixedSizeListArray, FixedSizeListVTable};
use crate::vtable::VisitorVTable;
use crate::{ArrayBufferVisitor, ArrayChildVisitor};

impl VisitorVTable<FixedSizeListVTable> for FixedSizeListVTable {
    fn visit_buffers(_array: &FixedSizeListArray, _visitor: &mut dyn ArrayBufferVisitor) {
        unimplemented!("TODO(connor)[FixedSizeList")
    }

    fn visit_children(array: &FixedSizeListArray, visitor: &mut dyn ArrayChildVisitor) {
        unimplemented!("TODO(connor)[FixedSizeList")
    }
}
