// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityChildSliceHelper;

use super::RLE;
use crate::RLEData;

impl ValidityChild<RLE> for RLE {
    fn validity_child(array: &RLEData) -> &ArrayRef {
        array.indices()
    }
}

impl ValidityChildSliceHelper for RLEData {
    fn unsliced_child_and_slice(&self) -> (&ArrayRef, usize, usize) {
        let (start, len) = (self.offset(), self.len());
        (self.indices(), start, start + len)
    }
}
