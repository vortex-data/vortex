// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Canonical;
use crate::arrays::VarBinViewVTable;
use crate::arrays::varbinview::VarBinViewArray;
use crate::builders::ArrayBuilder;
use crate::vtable::CanonicalVTable;

impl CanonicalVTable<VarBinViewVTable> for VarBinViewVTable {
    fn canonicalize(array: &VarBinViewArray) -> VortexResult<Canonical> {
        Ok(Canonical::VarBinView(array.clone()))
    }

    fn append_to_builder(
        array: &VarBinViewArray,
        builder: &mut dyn ArrayBuilder,
    ) -> VortexResult<()> {
        builder.extend_from_array(array.as_ref());
        Ok(())
    }
}
