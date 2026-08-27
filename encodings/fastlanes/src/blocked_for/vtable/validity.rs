// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::vtable::ValidityChild;

use super::BlockedFoR;
use crate::blocked_for::array::BlockedFoRArraySlotsExt;

impl ValidityChild<BlockedFoR> for BlockedFoR {
    fn validity_child(array: ArrayView<'_, BlockedFoR>) -> ArrayRef {
        array.encoded().clone()
    }
}
