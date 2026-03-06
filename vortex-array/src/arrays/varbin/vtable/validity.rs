// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::VarBinArray;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

impl ValidityHelper for VarBinArray {
    fn validity(&self) -> &Validity {
        &self.validity
    }
}
