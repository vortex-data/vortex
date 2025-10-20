// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::{Alignment, ByteBuffer};

use super::serde::BoolMetadata;
use crate::arrays::{BoolArray, BoolVTable};
use crate::vtable::{VTable, VisitorVTable};
use crate::{ArrayBufferVisitor, ArrayChildVisitor, ProstMetadata};

impl VisitorVTable<BoolVTable> for BoolVTable {
    fn metadata(array: &BoolArray) -> <BoolVTable as VTable>::Metadata {
        let bit_offset = array.boolean_buffer().offset();
        let bit_offset =
            u32::try_from(bit_offset).expect(&format!("bit_offset {bit_offset} overflows u32"));
        ProstMetadata(BoolMetadata { offset: bit_offset })
    }

    fn visit_buffers(array: &BoolArray, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(&ByteBuffer::from_arrow_buffer(
            array.boolean_buffer().clone().into_inner(),
            Alignment::none(),
        ))
    }

    fn visit_children(array: &BoolArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(&array.validity, array.len());
    }
}
