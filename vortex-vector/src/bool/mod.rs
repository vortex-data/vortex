// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`BoolVector`] and [`BoolVectorMut`].

mod vector;
pub use vector::BoolVector;

mod vector_mut;
pub use vector_mut::BoolVectorMut;

mod from_iter;

use crate::{Vector, VectorMut};

impl From<BoolVector> for Vector {
    fn from(v: BoolVector) -> Self {
        Self::Bool(v)
    }
}

impl From<BoolVectorMut> for VectorMut {
    fn from(v: BoolVectorMut) -> Self {
        Self::Bool(v)
    }
}
