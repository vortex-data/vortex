// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_mask::Mask;
use vortex_vector::primitive::{PrimitiveVector, PrimitiveVectorMut};
use vortex_vector::{match_each_pvector, match_each_pvector_mut};

use crate::filter::{Filter, MaskIndices};

impl Filter<Mask> for &PrimitiveVector {
    type Output = PrimitiveVector;

    fn filter(self, selection_mask: &Mask) -> PrimitiveVector {
        match_each_pvector!(self, |v| { v.filter(selection_mask).into() })
    }
}

impl Filter<MaskIndices<'_>> for &PrimitiveVector {
    type Output = PrimitiveVector;

    fn filter(self, indices: &MaskIndices<'_>) -> Self::Output {
        match_each_pvector!(self, |v| { v.filter(indices).into() })
    }
}

impl Filter<Mask> for &mut PrimitiveVectorMut {
    type Output = ();

    fn filter(self, selection_mask: &Mask) {
        match_each_pvector_mut!(self, |v| { v.filter(selection_mask) })
    }
}

impl Filter<MaskIndices<'_>> for &mut PrimitiveVectorMut {
    type Output = ();

    fn filter(self, indices: &MaskIndices<'_>) -> Self::Output {
        match_each_pvector_mut!(self, |v| { v.filter(indices) })
    }
}
