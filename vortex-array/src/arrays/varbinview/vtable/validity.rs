// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::array::ArrayView;
use crate::array::ValidityVTable;
use crate::arrays::varbinview::vtable::VarBinView;
use crate::validity::Validity;

impl ValidityVTable<VarBinView> for VarBinView {
    fn validity(array: ArrayView<'_, VarBinView>) -> VortexResult<Validity> {
        Ok(array.data().validity())
    }
}
