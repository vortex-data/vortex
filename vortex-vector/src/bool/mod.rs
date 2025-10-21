// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`BoolVector`] and [`BoolVectorMut`].

mod vector;
pub use vector::BoolVector;

mod vector_mut;
use crate::{Vector, VectorMut};
pub use vector_mut::BoolVectorMut;
use vortex_error::vortex_panic;

impl From<BoolVector> for Vector {
    fn from(v: BoolVector) -> Self {
        Self::Bool(v)
    }
}

impl From<Vector> for BoolVector {
    fn from(value: Vector) -> Self {
        if let Vector::Bool(v) = value {
            return v;
        }
        vortex_panic!("Expected BoolVector, got {value:?}");
    }
}

impl From<BoolVectorMut> for VectorMut {
    fn from(v: BoolVectorMut) -> Self {
        Self::Bool(v)
    }
}

impl From<VectorMut> for BoolVectorMut {
    fn from(value: VectorMut) -> Self {
        if let VectorMut::Bool(v) = value {
            return v;
        }
        vortex_panic!("Expected BoolVectorMut, got {value:?}");
    }
}
