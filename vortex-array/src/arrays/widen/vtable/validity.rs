// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::array::ValidityChild;
use crate::arrays::Widen;
use crate::arrays::widen::WidenArraySlotsExt;

impl ValidityChild<Widen> for Widen {
    fn validity_child(array: ArrayView<'_, Widen>) -> ArrayRef {
        array.values().clone()
    }
}
