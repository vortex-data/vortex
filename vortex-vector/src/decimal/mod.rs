// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definitions and implementations of decimal vector types.
//!
//! The types that hold data are [`DVector`] and [`DVectorMut`], which are generic over types `D`
//! that implement [`NativeDecimalType`].
//!
//! [`DecimalVector`] and [`DecimalVectorMut`] are enums that wrap all of the different possible
//! [`DVector`]s. There are several macros defined in this crate to make working with these
//! primitive vector types easier.
//!
//! # Examples
//!
//! ## Creating and building decimal vectors
//!
//! ```
//! use vortex_dtype::{PrecisionScale};
//! use vortex_vector::decimal::{DVectorMut};
//! use vortex_vector::VectorMutOps;
//!
//! // Create a decimal vector with precision=9, scale=2 (e.g., up to 9999999.99).
//! let ps = PrecisionScale::<i32>::new(9, 2);
//! let mut vec = DVectorMut::<i32>::with_capacity(ps, 5);
//! assert_eq!(vec.len(), 0);
//! assert!(vec.capacity() >= 5);
//!
//! // Values are stored as integers scaled by 10^scale.
//! // For scale=2: 123.45 is stored as 12345.
//! vec.try_push(12345).unwrap();  // Represents 123.45.
//! vec.try_push(9999).unwrap();   // Represents 99.99.
//! assert_eq!(vec.len(), 2);
//!
//! // Values that exceed precision will fail.
//! let too_large = 10_i32.pow(9);  // Would represent 10000000.00.
//! assert!(vec.try_push(too_large).is_err());
//!
//! // Create from buffers with validation.
//! use vortex_buffer::BufferMut;
//! use vortex_mask::MaskMut;
//! let elements = BufferMut::from_iter([100_i32, 200, 300]);  // 1.00, 2.00, 3.00.
//! let validity = MaskMut::new_true(3);
//! let decimal_vec = DVectorMut::<i32>::new(ps, elements, validity);
//! assert_eq!(decimal_vec.len(), 3);
//! ```
//!
//! ## Working with nulls and validity
//!
//! ```
//! use vortex_buffer::BufferMut;
//! use vortex_dtype::{PrecisionScale};
//! use vortex_mask::MaskMut;
//! use vortex_vector::decimal::DVectorMut;
//! use vortex_vector::VectorMutOps;
//!
//! // Create a decimal vector with nulls.
//! let ps = PrecisionScale::<i32>::new(5, 2); // Up to 999.99.
//!
//! // Create with some null values (validity mask: true = not null, false = null).
//! let elements = BufferMut::from_iter([1000_i32, 0, 2500, 0]);  // 10.00, null, 25.00, null.
//! let mut validity = MaskMut::with_capacity(4);
//! validity.append_n(true, 1);   // index 0: valid
//! validity.append_n(false, 1);  // index 1: null
//! validity.append_n(true, 1);   // index 2: valid
//! validity.append_n(false, 1);  // index 3: null
//! let mut vec = DVectorMut::new(ps, elements, validity);
//!
//! // Check element access with nulls.
//! assert_eq!(vec.get(0), Some(&1000));  // 10.00.
//! assert_eq!(vec.get(1), None);         // Null.
//! assert_eq!(vec.get(2), Some(&2500));  // 25.00.
//!
//! // Append null values.
//! vec.append_nulls(3);
//! assert_eq!(vec.len(), 7);
//! ```
//!
//! ## Extending and manipulating vectors
//!
//! ```
//! use vortex_dtype::{PrecisionScale};
//! use vortex_vector::decimal::DVectorMut;
//! use vortex_vector::VectorMutOps;
//!
//! // Create two decimal vectors with scale=3 (3 decimal places).
//! let ps = PrecisionScale::<i64>::new(10, 3);
//! let mut vec1 = DVectorMut::<i64>::with_capacity(ps, 10);
//! vec1.try_push(1234567).unwrap();  // 1234.567.
//! vec1.try_push(2345678).unwrap();  // 2345.678.
//!
//! let mut vec2 = DVectorMut::<i64>::with_capacity(ps, 10);
//! vec2.try_push(3456789).unwrap();  // 3456.789.
//! vec2.try_push(4567890).unwrap();  // 4567.890.
//!
//! // Extend from an immutable vector.
//! let immutable = vec2.freeze();
//! vec1.extend_from_vector(&immutable);
//! assert_eq!(vec1.len(), 4);
//!
//! // Split vector at index 3.
//! let mut split = vec1.split_off(3);
//! assert_eq!(vec1.len(), 3);
//! assert_eq!(split.len(), 1);
//!
//! // Reserve capacity for future operations.
//! vec1.reserve(10);
//! assert!(vec1.capacity() >= 13);
//!
//! // Rejoin the vectors.
//! vec1.unsplit(split);
//! assert_eq!(vec1.len(), 4);
//! ```
//!
//! ## Converting between mutable and immutable
//!
//! ```
//! use vortex_dtype::{PrecisionScale};
//! use vortex_vector::decimal::DVectorMut;
//! use vortex_vector::{VectorMutOps, VectorOps};
//!
//! // Create a mutable decimal vector.
//! let ps = PrecisionScale::<i128>::new(18, 6);  // High precision with 6 decimal places.
//! let mut vec_mut = DVectorMut::<i128>::with_capacity(ps, 3);
//! vec_mut.try_push(1000000).unwrap();    // 1.000000.
//! vec_mut.try_push(2500000).unwrap();    // 2.500000.
//! vec_mut.try_push(3333333).unwrap();    // 3.333333.
//!
//! // Freeze into an immutable vector.
//! let vec_immutable = vec_mut.freeze();
//! assert_eq!(vec_immutable.len(), 3);
//!
//! // Access elements from the immutable vector.
//! assert_eq!(vec_immutable.get(0), Some(&1000000));
//! assert_eq!(vec_immutable.get(1), Some(&2500000));
//!
//! // Can also convert immutable back to mutable using try_into_mut.
//! // Note: This may fail if the buffer is shared.
//! // let vec_mut_again = vec_immutable.try_into_mut().unwrap();
//! // assert_eq!(vec_mut_again.len(), 3);
//! ```

mod generic;
pub use generic::DVector;

mod generic_mut;
pub use generic_mut::DVectorMut;

mod vector;
pub use vector::DecimalVector;

mod vector_mut;
pub use vector_mut::DecimalVectorMut;

mod scalar;
pub use scalar::DScalar;
pub use scalar::DecimalScalar;

mod macros;

use vortex_dtype::NativeDecimalType;

use crate::Vector;
use crate::VectorMut;

impl From<DecimalVector> for Vector {
    fn from(v: DecimalVector) -> Self {
        Self::Decimal(v)
    }
}

impl<D: NativeDecimalType> From<DVector<D>> for DecimalVector {
    fn from(value: DVector<D>) -> Self {
        D::upcast(value)
    }
}

impl<D: NativeDecimalType> From<DVector<D>> for Vector {
    fn from(v: DVector<D>) -> Self {
        Self::Decimal(DecimalVector::from(v))
    }
}

impl From<DecimalVectorMut> for VectorMut {
    fn from(v: DecimalVectorMut) -> Self {
        Self::Decimal(v)
    }
}

impl<D: NativeDecimalType> From<DVectorMut<D>> for DecimalVectorMut {
    fn from(val: DVectorMut<D>) -> Self {
        D::upcast(val)
    }
}

impl<D: NativeDecimalType> From<DVectorMut<D>> for VectorMut {
    fn from(val: DVectorMut<D>) -> Self {
        Self::Decimal(DecimalVectorMut::from(val))
    }
}
