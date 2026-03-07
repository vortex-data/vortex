// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::vtable::ValidityChild;

use super::FoRVTable;
use crate::FoRArray;
use crate::FoRArrayExt;

impl ValidityChild<FoRVTable> for FoRVTable {
    fn validity_child(array: &FoRArray) -> &ArrayRef {
        array.encoded()
    }
}
