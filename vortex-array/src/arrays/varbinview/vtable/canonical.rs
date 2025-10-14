// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::Canonical;
use crate::arrays::VarBinViewVTable;
use crate::arrays::varbinview::VarBinViewArray;
use crate::builders::ArrayBuilder;
use crate::vtable::CanonicalVTable;

impl CanonicalVTable<VarBinViewVTable> for VarBinViewVTable {
    fn canonicalize(array: &VarBinViewArray) -> Canonical {
        Canonical::VarBinView(array.clone())
    }

    fn append_to_builder(array: &VarBinViewArray, builder: &mut dyn ArrayBuilder) {
        builder.extend_from_array(array.as_ref())
    }
}
