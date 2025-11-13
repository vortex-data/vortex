// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_mask::Mask;
use vortex_vector::{Vector, VectorMut, VectorMutOps, match_each_vector, match_each_vector_mut};

use crate::filter::{Filter, MaskIndices};

mod binaryview;
mod bool;
mod decimal;
mod dvector;
mod fixed_size_list;
mod list;
mod null;
mod primitive;
mod pvector;
mod struct_;

impl Filter<Mask> for &Vector {
    type Output = Vector;

    fn filter(self, selection: &Mask) -> Self::Output {
        match_each_vector!(self, |v| { v.filter(selection).into() })
    }
}

impl Filter<MaskIndices<'_>> for &Vector {
    type Output = Vector;

    fn filter(self, selection: &MaskIndices) -> Self::Output {
        match_each_vector!(self, |v| { v.filter(selection).into() })
    }
}

impl Filter<Mask> for &mut VectorMut {
    type Output = ();

    fn filter(self, selection_mask: &Mask) {
        match selection_mask {
            Mask::AllTrue(_) => {}
            Mask::AllFalse(_) => self.clear(),
            Mask::Values(_) => {
                match_each_vector_mut!(self, |v| { v.filter(selection_mask) });
            }
        }
    }
}

impl Filter<MaskIndices<'_>> for &mut VectorMut {
    type Output = ();

    fn filter(self, indices: &MaskIndices<'_>) -> Self::Output {
        match_each_vector_mut!(self, |v| { v.filter(indices) })
    }
}
