// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::vtable::VarBinView;
use crate::validity::Validity;
use crate::vtable::ValidityVTable;

impl ValidityVTable<VarBinView> for VarBinView {
    fn validity(array: &VarBinViewArray) -> VortexResult<Validity> {
        Ok(array.validity())
    }
}
