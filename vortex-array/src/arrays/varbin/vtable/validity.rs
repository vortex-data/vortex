// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::array::ArrayView;
use crate::array::ValidityVTable;
use crate::arrays::varbin::vtable::VarBin;
use crate::validity::Validity;

impl ValidityVTable<VarBin> for VarBin {
    fn validity(array: ArrayView<'_, VarBin>) -> VortexResult<Validity> {
        Ok(array.data().validity())
    }
}
