// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::Canonical;
use crate::arrays::{FixedSizeListArray, FixedSizeListVTable};
use crate::vtable::CanonicalVTable;

impl CanonicalVTable<FixedSizeListVTable> for FixedSizeListVTable {
    fn canonicalize(array: &FixedSizeListArray) -> Canonical {
        Canonical::FixedSizeList(array.clone())
    }
}
