// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_mask::Mask;
use vortex_vector::{Vector, VectorMut, match_each_vector, match_each_vector_mut};

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

// To allow all vector types to implement filter generically over `M`, we must break the recursive
// trait bounds (e.g. from StructVector requiring Vector: Filter<M> for its fields) by manually
// implementing Filter for Vector and VectorMut for each concrete mask type here.

macro_rules! impl_vector_filter {
    ($M:ty) => {
        impl Filter<$M> for &Vector {
            type Output = Vector;

            fn filter(self, selection: &$M) -> Self::Output {
                match_each_vector!(self, |v| { v.filter(selection).into() })
            }
        }

        impl Filter<$M> for &mut VectorMut {
            type Output = ();

            fn filter(self, selection: &$M) -> Self::Output {
                match_each_vector_mut!(self, |v| { v.filter(selection) })
            }
        }
    };
}

impl_vector_filter!(Mask);
impl_vector_filter!(MaskIndices<'_>);
