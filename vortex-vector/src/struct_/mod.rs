// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`StructVector`] and [`StructVectorMut`].
//!
//! # Examples
//!
//! ## Creating a [`StructVector`] and [`StructVectorMut`]
//!
//! ```
//! use vortex_vector::bool::BoolVectorMut;
//! use vortex_vector::null::NullVectorMut;
//! use vortex_vector::primitive::PVectorMut;
//! use vortex_vector::struct_::StructVectorMut;
//! use vortex_vector::{VectorMut, VectorMutOps};
//! use vortex_mask::MaskMut;
//!
//! // Create a struct with three fields: nulls, booleans, and integers.
//! let fields = Box::new([
//!     NullVectorMut::new(3).into(),
//!     BoolVectorMut::from_iter([true, false, true]).into(),
//!     PVectorMut::<i32>::from_iter([10, 20, 30]).into(),
//! ]);
//!
//! let mut struct_vec = StructVectorMut::new(fields, MaskMut::new_true(3));
//! assert_eq!(struct_vec.len(), 3);
//! ```
//!
//! ## Working with [`split_off()`] and [`unsplit()`]
//!
//! [`split_off()`]: crate::VectorMutOps::split_off
//! [`unsplit()`]: crate::VectorMutOps::unsplit
//!
//! ```
//! use vortex_vector::bool::BoolVectorMut;
//! use vortex_vector::null::NullVectorMut;
//! use vortex_vector::primitive::PVectorMut;
//! use vortex_vector::struct_::StructVectorMut;
//! use vortex_vector::{VectorMut, VectorMutOps};
//! use vortex_mask::MaskMut;
//!
//! let fields = Box::new([
//!     NullVectorMut::new(6).into(),
//!     PVectorMut::<i32>::from_iter([1, 2, 3, 4, 5, 6]).into(),
//! ]);
//!
//! let mut struct_vec = StructVectorMut::new(fields, MaskMut::new_true(6));
//!
//! // Split at position 4.
//! let second_part = struct_vec.split_off(4);
//!
//! assert_eq!(struct_vec.len(), 4);
//! assert_eq!(second_part.len(), 2);
//!
//! // Rejoin the parts.
//! struct_vec.unsplit(second_part);
//! assert_eq!(struct_vec.len(), 6);
//! ```
//!
//! ## Accessing field values
//!
//! ```
//! use vortex_vector::bool::BoolVectorMut;
//! use vortex_vector::null::NullVectorMut;
//! use vortex_vector::primitive::PVectorMut;
//! use vortex_vector::struct_::StructVectorMut;
//! use vortex_vector::{VectorMut, VectorMutOps};
//! use vortex_mask::MaskMut;
//! use vortex_dtype::PTypeDowncast;
//!
//! let fields = Box::new([
//!     NullVectorMut::new(3).into(),
//!     BoolVectorMut::from_iter([true, false, true]).into(),
//!     PVectorMut::<i32>::from_iter([10, 20, 30]).into(),
//! ]);
//!
//! let struct_vec = StructVectorMut::new(fields, MaskMut::new_true(3));
//!
//! // Access the boolean field vector (field index 1).
//! if let VectorMut::Bool(bool_vec) = struct_vec.fields()[1].clone() {
//!     let values: Vec<_> = bool_vec.into_iter().map(|v| v.unwrap()).collect();
//!     assert_eq!(values, vec![true, false, true]);
//! }
//!
//! // Access the integer field column (field index 2).
//! if let VectorMut::Primitive(prim_vec) = struct_vec.fields()[2].clone() {
//!     let values: Vec<_> = prim_vec.into_i32().into_iter().map(|v| v.unwrap()).collect();
//!     assert_eq!(values, vec![10, 20, 30]);
//! }
//! ```

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
