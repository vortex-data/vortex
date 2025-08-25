// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityHelper;

use crate::fsst_view::FSSTViewArray;

impl ValidityHelper for FSSTViewArray {
    fn validity(&self) -> &Validity {
        &self.validity
    }
}
