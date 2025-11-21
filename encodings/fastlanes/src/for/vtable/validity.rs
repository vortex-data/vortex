// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Array;
use vortex_array::vtable::ValidityChild;

use super::FoRVTable;
use crate::FoRArray;

impl ValidityChild<FoRVTable> for FoRVTable {
    fn validity_child(array: &FoRArray) -> &dyn Array {
        array.encoded().as_ref()
    }
}
