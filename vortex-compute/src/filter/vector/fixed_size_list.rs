// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_mask::Mask;
use vortex_vector::fixed_size_list::{FixedSizeListVector, FixedSizeListVectorMut};

use crate::filter::{Filter, MaskIndices};

// TODO(aduffy): there really isn't a cheap way to implement these is there.

impl Filter<Mask> for &FixedSizeListVector {
    type Output = FixedSizeListVector;

    fn filter(self, _selection: &Mask) -> Self::Output {
        // We need to spread the mask out to point to offsets from
        // the inner vector type
        todo!()
    }
}

impl Filter<MaskIndices<'_>> for &FixedSizeListVector {
    type Output = FixedSizeListVector;

    fn filter(self, _selection: &MaskIndices) -> Self::Output {
        // We need to spread the mask out to point to offsets from
        // the inner vector type
        todo!()
    }
}

impl Filter<Mask> for &mut FixedSizeListVectorMut {
    type Output = ();

    fn filter(self, _selection: &Mask) -> Self::Output {
        // We need to spread the mask out to point to offsets from
        // the inner vector type
        todo!()
    }
}

impl Filter<MaskIndices<'_>> for &mut FixedSizeListVectorMut {
    type Output = ();

    fn filter(self, _selection: &MaskIndices) -> Self::Output {
        // We need to spread the mask out to point to offsets from
        // the inner vector type
        todo!()
    }
}
