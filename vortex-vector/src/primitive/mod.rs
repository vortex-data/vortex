// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definitions and implementations of native primitive vector types.
//!
//! The types that hold data are [`GenericPVector`] and [`GenericPVectorMut`], which are generic
//! over types `T` that implement [`NativePType`] (which are just the integer and floating-point
//! types that are native to Rust plus [`f16`]).
//!
//! [`PrimitiveVector`] and [`PrimitiveVectorMut`] are enums that wrap all of the different possible
//! [`GenericPVector`]s. There are several macros defined in this crate to make working with these
//! primitive vector types easier.
//!
//! [`NativePType`]: vortex_dtype::NativePType
//! [`f16`]: vortex_dtype::half::f16

mod generic;
pub use generic::GenericPVector;

mod generic_mut;
pub use generic_mut::GenericPVectorMut;

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

impl<T: NativePType> From<GenericPVector<T>> for Vector {
    fn from(v: GenericPVector<T>) -> Self {
        Self::Primitive(PrimitiveVector::from(v))
    }
}

impl From<PrimitiveVectorMut> for VectorMut {
    fn from(v: PrimitiveVectorMut) -> Self {
        Self::Primitive(v)
    }
}

impl<T: NativePType> From<GenericPVectorMut<T>> for VectorMut {
    fn from(val: GenericPVectorMut<T>) -> Self {
        Self::Primitive(PrimitiveVectorMut::from(val))
    }
}
