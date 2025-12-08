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
//! # Examples
//!
//! ## Creating and building a vector
//!
//! ```
//! use vortex_vector::primitive::PVectorMut;
//! use vortex_vector::VectorMutOps;
//!
//! // Create with initial capacity for i32 values.
//! let mut vec = PVectorMut::<i32>::with_capacity(10);
//! assert_eq!(vec.len(), 0);
//! assert!(vec.capacity() >= 10);
//!
//! // Create from an iterator of optional values.
//! let mut vec = PVectorMut::<i32>::from_iter([Some(1), None, Some(3)]);
//! assert_eq!(vec.len(), 3);
//!
//! // Works with different primitive types.
//! let mut f64_vec = PVectorMut::<f64>::from_iter([1.5, 2.5, 3.5].map(Some));
//! assert_eq!(f64_vec.len(), 3);
//! ```
//!
//! ## Extending and appending
//!
//! ```
//! use vortex_vector::primitive::PVectorMut;
//! use vortex_vector::VectorMutOps;
//!
//! let mut vec1 = PVectorMut::<i32>::from_iter([1, 2].map(Some));
//! let vec2 = PVectorMut::<i32>::from_iter([3, 4].map(Some)).freeze();
//!
//! // Extend from another vector.
//! vec1.extend_from_vector(&vec2);
//! assert_eq!(vec1.len(), 4);
//!
//! // Append null values.
//! vec1.append_nulls(2);
//! assert_eq!(vec1.len(), 6);
//! ```
//!
//! ## Splitting and unsplitting
//!
//! ```
//! use vortex_vector::primitive::PVectorMut;
//! use vortex_vector::VectorMutOps;
//!
//! let mut vec = PVectorMut::<i64>::from_iter([10, 20, 30, 40, 50].map(Some));
//!
//! // Split the vector at index 3.
//! let mut second_half = vec.split_off(3);
//! assert_eq!(vec.len(), 3);
//! assert_eq!(second_half.len(), 2);
//!
//! // Rejoin the vectors.
//! vec.unsplit(second_half);
//! assert_eq!(vec.len(), 5);
//! ```
//!
//! ## Working with nulls
//!
//! ```
//! use vortex_vector::primitive::PVectorMut;
//! use vortex_vector::VectorMutOps;
//!
//! // Create a vector with some null values.
//! let mut vec = PVectorMut::<u32>::from_iter([Some(100), None, Some(200), None]);
//! assert_eq!(vec.len(), 4);
//!
//! // Add more nulls.
//! vec.append_nulls(3);
//! assert_eq!(vec.len(), 7);
//! ```
//!
//! ## Converting to immutable
//!
//! ```
//! use vortex_vector::primitive::PVectorMut;
//! use vortex_vector::{VectorMutOps, VectorOps};
//!
//! let mut vec = PVectorMut::<f32>::from_iter([1.0, 2.0, 3.0].map(Some));
//!
//! // Freeze into an immutable vector.
//! let immutable = vec.freeze();
//! assert_eq!(immutable.len(), 3);
//! ```
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

mod scalar;
pub use scalar::PScalar;
pub use scalar::PrimitiveScalar;

mod macros;

use vortex_dtype::NativePType;

use crate::Vector;
use crate::VectorMut;

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
