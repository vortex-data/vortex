// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::vtable::ValidityChild;

use super::Affine;
use crate::affine::array::AffineArraySlotsExt;

impl ValidityChild<Affine> for Affine {
    fn validity_child(array: ArrayView<'_, Affine>) -> ArrayRef {
        array.encoded().clone()
    }
}
