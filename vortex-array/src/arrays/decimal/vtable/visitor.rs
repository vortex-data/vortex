// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use super::serde::DecimalMetadata;
use crate::arrays::{DecimalArray, DecimalVTable};
use crate::vtable::{VTable, ValidityHelper, VisitorVTable};
use crate::{ArrayBufferVisitor, ArrayChildVisitor, ProstMetadata};

impl VisitorVTable<DecimalVTable> for DecimalVTable {
    fn metadata(array: &DecimalArray) -> <DecimalVTable as VTable>::Metadata {
        ProstMetadata(DecimalMetadata {
            values_type: array.values_type() as i32,
        })
    }

    fn visit_buffers(array: &DecimalArray, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(&array.values);
    }

    fn visit_children(array: &DecimalArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_validity(array.validity(), array.len())
    }
}
