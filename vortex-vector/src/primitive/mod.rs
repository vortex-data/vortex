// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definitions and implementations of native primitive vector types.
//!
//! The types that hold data are [`PVector`] and [`PVectorMut`], which are generic over types `T`
//! that implement [`NativePType`] (which are just the integer and floating-point types that are
//! native to Rust plus [`f16`]).
//!
//! [`PrimitiveVector`] and [`PrimitiveVectorMut`] are enums that wrap all of the different possible
//! [`PVector`]s. There are several macros defined in this crate to make working with these
//! primitive vector types easier.
//!
//! [`NativePType`]: vortex_dtype::NativePType
//! [`f16`]: vortex_dtype::half::f16

mod generic;
pub use generic::PVector;

mod generic_mut;
pub use generic_mut::PVectorMut;

mod vector;
pub use vector::PrimitiveVector;

mod vector_mut;
pub use vector_mut::PrimitiveVectorMut;

mod macros;

use crate::{Vector, VectorMut};
use vortex_dtype::NativePType;
use vortex_error::vortex_panic;

impl From<PrimitiveVector> for Vector {
    fn from(v: PrimitiveVector) -> Self {
        Self::Primitive(v)
    }
}

impl From<Vector> for PrimitiveVector {
    fn from(value: Vector) -> Self {
        if let Vector::Primitive(v) = value {
            return v;
        }
        vortex_panic!("Expected PrimitiveVector, got {value:?}");
    }
}

impl<T: NativePType> From<PVector<T>> for Vector {
    fn from(v: PVector<T>) -> Self {
        Self::Primitive(PrimitiveVector::from(v))
    }
}

impl<T: NativePType> From<Vector> for PVector<T> {
    fn from(value: Vector) -> Self {
        if let Vector::Primitive(v) = value {
            return PVector::from(v);
        }
        vortex_panic!("Expected PrimitiveVector, got {value:?}");
    }
}

impl From<PrimitiveVectorMut> for VectorMut {
    fn from(v: PrimitiveVectorMut) -> Self {
        Self::Primitive(v)
    }
}

impl From<VectorMut> for PrimitiveVectorMut {
    fn from(value: VectorMut) -> Self {
        if let VectorMut::Primitive(v) = value {
            return v;
        }
        vortex_panic!("Expected PrimitiveVectorMut, got {value:?}");
    }
}

impl<T: NativePType> From<PVectorMut<T>> for VectorMut {
    fn from(val: PVectorMut<T>) -> Self {
        Self::Primitive(PrimitiveVectorMut::from(val))
    }
}

impl<T: NativePType> From<VectorMut> for PVectorMut<T> {
    fn from(value: VectorMut) -> Self {
        if let VectorMut::Primitive(v) = value {
            return PVectorMut::from(v);
        }
        vortex_panic!("Expected PrimitiveVectorMut, got {value:?}");
    }
}
