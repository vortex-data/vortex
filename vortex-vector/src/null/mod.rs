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

impl From<Vector> for NullVector {
    fn from(value: Vector) -> Self {
        if let Vector::Null(v) = value {
            return v;
        }
        panic!("Expected NullVector, got {value:?}");
    }
}

impl From<NullVectorMut> for VectorMut {
    fn from(v: NullVectorMut) -> Self {
        Self::Null(v)
    }
}

impl From<VectorMut> for NullVectorMut {
    fn from(value: VectorMut) -> Self {
        if let VectorMut::Null(v) = value {
            return v;
        }
        panic!("Expected NullVectorMut, got {value:?}");
    }
}
