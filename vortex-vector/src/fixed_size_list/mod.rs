// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`FixedSizeListVector`] and [`FixedSizeListVectorMut`].
//!
//! # Examples
//!
//! ## Working with nulls
//!
//! Nulls can exist at two levels: entire lists can be null, or individual elements within lists can
//! be null.
//!
//! ```
//! use vortex_vector::fixed_size_list::FixedSizeListVectorMut;
//! use vortex_vector::primitive::PVectorMut;
//! use vortex_vector::{VectorMut, VectorMutOps};
//! use vortex_mask::{Mask, MaskMut};
//!
//! // Create elements with some null values.
//! // This will be 9 elements total: [1, null, 3, 4, 5, null, null, 8, 9]
//! let mut elements = PVectorMut::<i32>::from_iter([
//!     Some(1), None, Some(3),       // First list
//!     Some(4), Some(5), None,       // Second list
//!     None, Some(8), Some(9),       // Third list
//! ]);
//!
//! // Create validity for the lists themselves.
//! // All lists are valid in this example.
//! let validity = MaskMut::new_true(3);
//!
//! let mut fsl_vec = FixedSizeListVectorMut::new(
//!     Box::new(elements.into()),
//!     3, // Each list has 3 elements
//!     validity,
//! );
//!
//! assert_eq!(fsl_vec.len(), 3);
//! assert_eq!(fsl_vec.list_size(), 3);
//!
//! // Can also append null lists.
//! fsl_vec.append_nulls(2);
//! assert_eq!(fsl_vec.len(), 5);
//! ```
//!
//! ## Working with [`split_off()`] and [`unsplit()`]
//!
//! [`split_off()`]: crate::VectorMutOps::split_off
//! [`unsplit()`]: crate::VectorMutOps::unsplit
//!
//! ```
//! use vortex_vector::fixed_size_list::FixedSizeListVectorMut;
//! use vortex_vector::primitive::PVectorMut;
//! use vortex_vector::{VectorMut, VectorMutOps};
//! use vortex_mask::MaskMut;
//!
//! // Create a vector with 6 lists, each containing 2 integers.
//! let elements = PVectorMut::<i32>::from_iter([
//!     1, 2,    // List 0
//!     3, 4,    // List 1
//!     5, 6,    // List 2
//!     7, 8,    // List 3
//!     9, 10,   // List 4
//!     11, 12,  // List 5
//! ]);
//!
//! let mut fsl_vec = FixedSizeListVectorMut::new(
//!     Box::new(elements.into()),
//!     2, // Each list has 2 elements
//!     MaskMut::new_true(6),
//! );
//!
//! // Split at position 4 (keeping first 4 lists, splitting off last 2).
//! let second_part = fsl_vec.split_off(4);
//!
//! assert_eq!(fsl_vec.len(), 4);
//! assert_eq!(second_part.len(), 2);
//!
//! // The elements are also split accordingly.
//! assert_eq!(fsl_vec.elements().len(), 8);  // 4 lists * 2 elements
//! assert_eq!(second_part.elements().len(), 4);  // 2 lists * 2 elements
//!
//! // Rejoin the parts.
//! fsl_vec.unsplit(second_part);
//! assert_eq!(fsl_vec.len(), 6);
//! assert_eq!(fsl_vec.elements().len(), 12);
//! ```

mod vector;
pub use vector::FixedSizeListVector;

mod scalar;
pub use scalar::FixedSizeListScalar;

mod vector_mut;
pub use vector_mut::FixedSizeListVectorMut;

use crate::Vector;
use crate::VectorMut;

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
