// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Canonical;
use crate::arrays::{FixedSizeListArray, FixedSizeListVTable};
use crate::vtable::CanonicalVTable;

impl CanonicalVTable<FixedSizeListVTable> for FixedSizeListVTable {
    fn canonicalize(array: &FixedSizeListArray) -> VortexResult<Canonical> {
        Ok(Canonical::FixedSizeList(array.clone()))
    }
}
