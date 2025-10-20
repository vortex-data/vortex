// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::PType;
use vortex_error::VortexExpect;

use super::serde::VarBinMetadata;
use crate::arrays::{VarBinArray, VarBinVTable};
use crate::vtable::{VTable, ValidityHelper, VisitorVTable};
use crate::{ArrayBufferVisitor, ArrayChildVisitor, ProstMetadata};

impl VisitorVTable<VarBinVTable> for VarBinVTable {
    fn metadata(array: &VarBinArray) -> <VarBinVTable as VTable>::Metadata {
        ProstMetadata(VarBinMetadata {
            offsets_ptype: PType::try_from(array.offsets().dtype())
                .vortex_expect("Must be a valid PType") as i32,
        })
    }

    fn visit_buffers(array: &VarBinArray, visitor: &mut dyn ArrayBufferVisitor) {
        visitor.visit_buffer(array.bytes()); // TODO(ngates): sliced bytes?
    }

    fn visit_children(array: &VarBinArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("offsets", array.offsets());
        visitor.visit_validity(array.validity(), array.len());
    }
}
