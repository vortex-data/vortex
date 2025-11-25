// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityChildSliceHelper;

use super::RLEVTable;
use crate::RLEArray;

impl ValidityChild<RLEVTable> for RLEVTable {
    fn validity_child(array: &RLEArray) -> &dyn Array {
        array.indices().as_ref()
    }
}

impl ValidityChildSliceHelper for RLEArray {
    fn unsliced_child_and_slice(&self) -> (&ArrayRef, usize, usize) {
        let (start, len) = (self.offset(), self.len());
        (self.indices(), start, start + len)
    }
}
