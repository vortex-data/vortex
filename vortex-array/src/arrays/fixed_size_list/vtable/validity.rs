// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::fixed_size_list::FixedSizeListData;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

impl ValidityHelper for FixedSizeListData {
    fn validity(&self) -> &Validity {
        &self.validity
    }
}
