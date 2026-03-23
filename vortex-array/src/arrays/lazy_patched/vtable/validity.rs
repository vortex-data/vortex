// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayRef;
use crate::array::ValidityChild;
use crate::arrays::lazy_patched::LazyPatched;
use crate::arrays::lazy_patched::LazyPatchedData;

impl ValidityChild<LazyPatched> for LazyPatched {
    fn validity_child(array: &LazyPatchedData) -> &ArrayRef {
        array.inner()
    }
}
