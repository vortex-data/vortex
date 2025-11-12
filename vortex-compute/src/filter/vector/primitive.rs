// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_mask::Mask;
use vortex_vector::primitive::{PrimitiveVector, PrimitiveVectorMut};
use vortex_vector::{VectorOps, match_each_pvector, match_each_pvector_mut};

use crate::filter::Filter;

impl Filter for &PrimitiveVector {
    type Output = PrimitiveVector;

    fn filter(self, selection_mask: &Mask) -> PrimitiveVector {
        match_each_pvector!(self, |v| { v.filter(selection_mask).into() })
    }
}

impl Filter for &mut PrimitiveVectorMut {
    type Output = ();

    fn filter(self, selection_mask: &Mask) {
        match_each_pvector_mut!(self, |v| { v.filter(selection_mask) })
    }
}

impl Filter for PrimitiveVector {
    type Output = Self;

    fn filter(self, selection_mask: &Mask) -> Self {
        assert_eq!(
            selection_mask.len(),
            self.len(),
            "Selection mask length must equal the buffer length"
        );

        match_each_pvector!(self, |v| { v.filter(selection_mask).into() })
    }
}
