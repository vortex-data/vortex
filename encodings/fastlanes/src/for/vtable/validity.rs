// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::vtable::ValidityChild;

use super::FoR;
use crate::FoRArray;

impl ValidityChild<FoR> for FoR {
    fn validity_child(array: &FoRArray) -> &ArrayRef {
        array.encoded()
    }
}
