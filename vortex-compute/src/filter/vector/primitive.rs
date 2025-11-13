// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_mask::Mask;
use vortex_vector::match_each_pvector_mut;
use vortex_vector::primitive::PrimitiveVectorMut;

use crate::filter::Filter;

impl Filter for &mut PrimitiveVectorMut {
    type Output = ();

    fn filter(self, selection_mask: &Mask) {
        match_each_pvector_mut!(self, |v| { v.filter(selection_mask) })
    }
}
