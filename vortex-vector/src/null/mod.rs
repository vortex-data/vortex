// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`NullVector`] and [`NullVectorMut`].

mod vector;
pub use vector::NullVector;

mod vector_mut;
pub use vector_mut::NullVectorMut;

use crate::{Vector, VectorMut};

impl From<NullVector> for Vector {
    fn from(v: NullVector) -> Self {
        Self::Null(v)
    }
}

impl From<NullVectorMut> for VectorMut {
    fn from(v: NullVectorMut) -> Self {
        Self::Null(v)
    }
}
