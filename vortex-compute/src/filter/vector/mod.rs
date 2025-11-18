// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitView;
use vortex_mask::Mask;
use vortex_vector::{Vector, VectorMut, match_each_vector, match_each_vector_mut};

use crate::filter::Filter;

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

// To allow all vector types to implement filter generically over `M`, we must break the recursive
// trait bounds (e.g. from StructVector requiring Vector: Filter<M> for its fields) by manually
// implementing Filter for Vector and VectorMut for each concrete mask type here.

impl Filter<Mask> for &Vector {
    type Output = Vector;

    fn filter(self, selection: &Mask) -> Self::Output {
        match_each_vector!(self, |v| { v.filter(selection).into() })
    }
}

impl Filter<Mask> for &mut VectorMut {
    type Output = ();

    fn filter(self, selection: &Mask) -> Self::Output {
        match_each_vector_mut!(self, |v| { v.filter(selection) })
    }
}

impl<const NB: usize> Filter<BitView<'_, NB>> for &Vector {
    type Output = Vector;

    fn filter(self, selection: &BitView<'_, NB>) -> Self::Output {
        match_each_vector!(self, |v| { v.filter(selection).into() })
    }
}

impl<const NB: usize> Filter<BitView<'_, NB>> for &mut VectorMut {
    type Output = ();

    fn filter(self, selection: &BitView<'_, NB>) -> Self::Output {
        match_each_vector_mut!(self, |v| { v.filter(selection) })
    }
}
