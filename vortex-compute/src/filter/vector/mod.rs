// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_mask::Mask;
use vortex_vector::{VectorMut, VectorMutOps};

use crate::filter::Filter;

mod bool;
mod primitive;
mod pvector;

impl Filter<Mask> for &mut VectorMut {
    type Output = ();

    fn filter(self, selection_mask: &Mask) {
        match selection_mask {
            Mask::AllTrue(_) => {}
            Mask::AllFalse(_) => self.clear(),
            Mask::Values(_) => {
                // match_each_vector_mut!(self, |v| { v.filter(selection_mask)
                todo!()
            }
        }
    }
}
