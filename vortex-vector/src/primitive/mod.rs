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
//! [`f16`]: vortex_dtype::half::f16

mod generic;
pub use generic::PVector;

mod generic_mut;
mod generic_mut_impl;
mod iter;
pub use generic_mut::PVectorMut;

mod vector;
pub use vector::PrimitiveVector;

mod vector_mut;
pub use vector_mut::PrimitiveVectorMut;

mod macros;

use vortex_dtype::NativePType;

use crate::{Vector, VectorMut};

impl From<PrimitiveVector> for Vector {
    fn from(v: PrimitiveVector) -> Self {
        Self::Primitive(v)
    }
}

impl<T: NativePType> From<PVector<T>> for PrimitiveVector {
    fn from(v: PVector<T>) -> Self {
        T::upcast(v)
    }
}

impl<T: NativePType> From<PVector<T>> for Vector {
    fn from(v: PVector<T>) -> Self {
        Self::Primitive(PrimitiveVector::from(v))
    }
}

impl From<PrimitiveVectorMut> for VectorMut {
    fn from(v: PrimitiveVectorMut) -> Self {
        Self::Primitive(v)
    }
}

impl<T: NativePType> From<PVectorMut<T>> for PrimitiveVectorMut {
    fn from(v: PVectorMut<T>) -> Self {
        T::upcast(v)
    }
}

impl<T: NativePType> From<PVectorMut<T>> for VectorMut {
    fn from(val: PVectorMut<T>) -> Self {
        Self::Primitive(PrimitiveVectorMut::from(val))
    }
}
