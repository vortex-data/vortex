// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`StructVector`] and [`StructVectorMut`].

mod vector;
pub use vector::StructVector;

mod vector_mut;
pub use vector_mut::StructVectorMut;

use crate::{Vector, VectorMut};

impl From<StructVector> for Vector {
    fn from(v: StructVector) -> Self {
        Self::Struct(v)
    }
}

impl From<StructVectorMut> for VectorMut {
    fn from(v: StructVectorMut) -> Self {
        Self::Struct(v)
    }
}
