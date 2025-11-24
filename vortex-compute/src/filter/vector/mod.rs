// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Filter implementations for vector types.
//!
//! This module provides [`Filter`] trait implementations for all vector types with three distinct
//! patterns optimized for different ownership scenarios:
//!
//! ### 1. Reference Implementation (`&Vector`)
//!
//! Filters by allocating new memory and copying selected elements. Returns a new owned vector.
//! This is the base implementation that all other patterns can fall back to.
//!
//! ### 2. Mutable Reference Implementation (`&mut VectorMut`)
//!
//! Filters in-place when exclusive mutable access is available, avoiding allocation. Returns `()`
//! as the vector is modified directly. This is the most efficient when you already have a mutable
//! vector (it is only less efficient if the vector is very small and the output vector is already
//! in the L1 cache).
//!
//! ### 3. Owned Implementation (`Vector`)
//!
//! Uses [`VectorOps::try_into_mut`] to check for exclusive ownership. If successful, performs
//! in-place filtering via the mutable implementation and calls [`VectorMutOps::freeze`] to convert
//! back. Otherwise, delegates to the reference implementation and makes an allocation.
//!
//! ## Breaking Recursive Trait Bounds
//!
//! To allow all vector types to implement filter generically over a "mask" type `M`, we must break
//! the recursive trait bounds (e.g. from [`StructVector`] requiring `Vector: Filter<M>` for its
//! fields) by manually implementing [`Filter`] for [`Vector`] and [`VectorMut`] for each concrete
//! mask type in this file.

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

// We manually implement Filter for Vector and VectorMut for each concrete mask type here to break
// the recursive trait bounds.

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
