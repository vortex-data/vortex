// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::Variant;
use crate::validity::Validity;
use crate::vtable::Array;
use crate::vtable::ValidityVTable;

impl ValidityVTable<Variant> for Variant {
    fn validity(array: &Array<Variant>) -> VortexResult<Validity> {
        array.child().validity()
    }
}
