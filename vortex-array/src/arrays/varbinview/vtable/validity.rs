// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::VarBinViewArray;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

impl ValidityHelper for VarBinViewArray {
    fn validity(&self) -> &Validity {
        &self.validity
    }
}
