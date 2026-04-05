// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::array::ValidityChild;
use crate::arrays::Extension;

impl ValidityChild<Extension> for Extension {
    fn validity_child(array: ArrayView<'_, Extension>) -> ArrayRef {
        array.storage_array().clone()
    }
}
