// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::varbinview::VarBinViewData;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

impl ValidityHelper for VarBinViewData {
    fn validity(&self) -> &Validity {
        &self.validity
    }
}
