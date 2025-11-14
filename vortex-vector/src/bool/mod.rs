// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`BoolVector`] and [`BoolVector`].
//!
//! # Examples
//!
//! ## Extending and appending
//!
//! ```
//! use vortex_vector::bool::BoolVector;
//! use vortex_vector::VectorOps;
//!
//! let mut vec1 = BoolVector::from_iter([true, false].map(Some));
//! let vec2 = BoolVector::from_iter([true, true].map(Some)).freeze();
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
//! use vortex_vector::bool::BoolVector;
//! use vortex_vector::VectorOps;
//!
//! let mut vec = BoolVector::from_iter([true, false, true, false, true].map(Some));
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
//! ## Converting to immutable
//!
//! ```
//! use vortex_vector::bool::BoolVector;
//! use vortex_vector::{VectorOps, VectorOps};
//!
//! let mut vec = BoolVector::from_iter([true, false, true].map(Some));
//!
//! // Freeze into an immutable vector.
//! let immutable = vec.freeze();
//! assert_eq!(immutable.len(), 3);
//! ```

mod vector_mut;
pub use vector_mut::BoolVector;

mod scalar;
pub use scalar::BoolScalar;

mod iter;

use crate::Vector;

impl From<BoolVector> for Vector {
    fn from(v: BoolVector) -> Self {
        Self::Bool(v)
    }
}
