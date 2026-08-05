// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::array::ValidityChild;
use crate::arrays::Map;
use crate::arrays::map::MapArraySlotsExt;

impl ValidityChild<Map> for Map {
    fn validity_child(array: ArrayView<'_, Map>) -> ArrayRef {
        array.entries().clone()
    }
}
