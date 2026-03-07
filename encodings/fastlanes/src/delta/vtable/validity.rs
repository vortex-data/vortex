// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::vtable::ValidityChildSliceHelper;

use crate::DeltaArray;
use crate::DeltaArrayExt;

impl ValidityChildSliceHelper for DeltaArray {
    fn unsliced_child_and_slice(&self) -> (&ArrayRef, usize, usize) {
        let (start, len) = (self.offset(), self.len());
        (self.deltas(), start, start + len)
    }
}
