// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::vtable::VisitorVTable;
use vortex_array::{ArrayBufferVisitor, ArrayChildVisitor};

use crate::fsst_view::{FSSTViewArray, FSSTViewVTable};

impl VisitorVTable<FSSTViewVTable> for FSSTViewVTable {
    fn visit_buffers(array: &FSSTViewArray, visitor: &mut dyn ArrayBufferVisitor) {
        // Access the Views and FSST buffers.
        visitor.visit_buffer(&array.views.clone().into_byte_buffer());
        visitor.visit_buffer(&array.symbols.clone().into_byte_buffer());
        visitor.visit_buffer(&array.symbol_lengths);
        visitor.visit_buffer(&array.fsst_buffer);
    }

    fn visit_children(array: &FSSTViewArray, visitor: &mut dyn ArrayChildVisitor) {
        // Child arrays: uncompressed offsets, compressed offsets, validity (if present)
        visitor.visit_child("compressed_offsets", &array.compressed_offsets);
        visitor.visit_child("uncompressed_offsets", &array.uncompressed_offsets);
        visitor.visit_validity(&array.validity, array.len());
    }
}
