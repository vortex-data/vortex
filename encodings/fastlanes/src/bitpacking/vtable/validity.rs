// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityHelper;

use crate::BitPackedArray;

impl ValidityHelper for BitPackedArray {
    fn validity(&self) -> &Validity {
        &self.validity
    }
}
