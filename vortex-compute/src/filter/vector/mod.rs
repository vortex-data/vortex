// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_mask::Mask;
use vortex_vector::{VectorMut, VectorMutOps};

use crate::filter::Filter;

mod bool;
mod primitive;
mod pvector;

impl Filter for &mut VectorMut {
    type Output = ();

    fn filter(self, selection_mask: &Mask) {
        match selection_mask {
            Mask::AllTrue(_) => {}
            Mask::AllFalse(_) => self.clear(),
            Mask::Values(_) => {
                if let VectorMut::Primitive(primitive) = self {
                    primitive.filter(selection_mask);
                    return;
                }

                // match_each_vector_mut!(self, |v| { v.filter(selection_mask) })

                unimplemented!("Filter has not been implemented for all vectors")
            }
        }
    }
}
