// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`StructVector`] and [`StructVector`].
//!
//! # Examples
//!
//! ## Creating a [`StructVector`] and [`StructVector`]
//!
//! ```
//! use vortex_vector::bool::BoolVector;
//! use vortex_vector::null::NullVector;
//! use vortex_vector::primitive::PVector;
//! use vortex_vector::struct_::StructVector;
//! use vortex_vector::{Vector, VectorOps};
//! use vortex_mask::MaskMut;
//!
//! // Create a struct with three fields: nulls, booleans, and integers.
//! let fields = Box::new([
//!     NullVector::new(3).into(),
//!     BoolVector::from_iter([true, false, true]).into(),
//!     PVector::<i32>::from_iter([10, 20, 30]).into(),
//! ]);
//!
//! let mut struct_vec = StructVector::new(fields, MaskMut::new_true(3));
//! assert_eq!(struct_vec.len(), 3);
//! ```
//!
//! ## Working with [`split_off()`] and [`unsplit()`]
//!
//! [`split_off()`]: crate::VectorOps::split_off
//! [`unsplit()`]: crate::VectorOps::unsplit
//!
//! ```
//! use vortex_vector::bool::BoolVector;
//! use vortex_vector::null::NullVector;
//! use vortex_vector::primitive::PVector;
//! use vortex_vector::struct_::StructVector;
//! use vortex_vector::{Vector, VectorOps};
//! use vortex_mask::MaskMut;
//!
//! let fields = Box::new([
//!     NullVector::new(6).into(),
//!     PVector::<i32>::from_iter([1, 2, 3, 4, 5, 6]).into(),
//! ]);
//!
//! let mut struct_vec = StructVector::new(fields, MaskMut::new_true(6));
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
//! use vortex_vector::bool::BoolVector;
//! use vortex_vector::null::NullVector;
//! use vortex_vector::primitive::PVector;
//! use vortex_vector::struct_::StructVector;
//! use vortex_vector::{Vector, VectorOps};
//! use vortex_mask::MaskMut;
//! use vortex_dtype::PTypeDowncast;
//!
//! let fields = Box::new([
//!     NullVector::new(3).into(),
//!     BoolVector::from_iter([true, false, true]).into(),
//!     PVector::<i32>::from_iter([10, 20, 30]).into(),
//! ]);
//!
//! let struct_vec = StructVector::new(fields, MaskMut::new_true(3));
//!
//! // Access the boolean field vector (field index 1).
//! if let Vector::Bool(bool_vec) = struct_vec.fields()[1].clone() {
//!     let values: Vec<_> = bool_vec.into_iter().map(|v| v.unwrap()).collect();
//!     assert_eq!(values, vec![true, false, true]);
//! }
//!
//! // Access the integer field column (field index 2).
//! if let Vector::Primitive(prim_vec) = struct_vec.fields()[2].clone() {
//!     let values: Vec<_> = prim_vec.into_i32().into_iter().map(|v| v.unwrap()).collect();
//!     assert_eq!(values, vec![10, 20, 30]);
//! }
//! ```

mod vector_mut;
pub use vector_mut::StructVector;

mod scalar;

pub use scalar::StructScalar;

use crate::Vector;

impl From<StructVector> for Vector {
    fn from(v: StructVector) -> Self {
        Self::Struct(v)
    }
}
