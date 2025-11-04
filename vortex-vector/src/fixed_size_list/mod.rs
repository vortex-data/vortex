// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`FixedSizeListVector`] and [`FixedSizeListVectorMut`].

mod vector;
pub use vector::FixedSizeListVector;

mod vector_mut;
pub use vector_mut::FixedSizeListVectorMut;

use crate::{Vector, VectorMut};

impl From<FixedSizeListVector> for Vector {
    fn from(v: FixedSizeListVector) -> Self {
        Self::FixedSizeList(v)
    }
}

impl From<FixedSizeListVectorMut> for VectorMut {
    fn from(v: FixedSizeListVectorMut) -> Self {
        Self::FixedSizeList(v)
    }
}
