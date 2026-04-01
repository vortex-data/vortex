// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::VarBinArray;
use crate::arrays::varbin::vtable::VarBin;
use crate::validity::Validity;
use crate::vtable::ValidityVTable;

impl ValidityVTable<VarBin> for VarBin {
    fn validity(array: &VarBinArray) -> VortexResult<Validity> {
        Ok(array.validity())
    }
}
