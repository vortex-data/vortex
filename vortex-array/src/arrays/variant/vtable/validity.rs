// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::Variant;
use crate::validity::Validity;
use crate::vtable::ArrayView;
use crate::vtable::ValidityVTable;

impl ValidityVTable<Variant> for Variant {
    fn validity(array: ArrayView<'_, Variant>) -> VortexResult<Validity> {
        array.child().validity()
    }
}
