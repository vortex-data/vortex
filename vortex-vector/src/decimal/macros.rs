// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Helper macros for working with the different variants of [`DecimalVector`] and
//! [`DecimalVectorMut`].
//!
//! [`DecimalVector`]: crate::DecimalVector
//! [`DecimalVectorMut`]: crate::DecimalVectorMut

/// Matches on all decimal type variants of [`DecimalVector`] and executes the same code for
/// each variant branch.
///
/// This macro eliminates repetitive match statements when implementing operations that need to work
/// uniformly across all decimal type variants (`D8`, `D16`, `D32`, `D64`, `D128`, `D256`).
///
/// # Examples
///
/// ```
/// use vortex_vector::{DecimalVector, DVectorMut, VectorOps, VectorMutOps, match_each_dvector};
///
/// fn get_decimal_len(vector: &DecimalVector) -> usize {
///     match_each_dvector!(vector, |v| { v.len() })
/// }
///
/// // Works with `D32` decimal vectors.
/// let d32_vec: DecimalVector = DVectorMut::<i32>::from_iter([1, 2, 3].map(Some))
///     .freeze()
///     .into();
/// assert_eq!(get_decimal_len(&d32_vec), 3);
///
/// // Works with `D128` decimal vectors.
/// let d128_vec: DecimalVector = DVectorMut::<i128>::from_iter([100, 200].map(Some))
///     .freeze()
///     .into();
/// assert_eq!(get_decimal_len(&d128_vec), 2);
/// ```
///
/// Note: The `len` method is already provided by the [`VectorOps`] trait implementation.
///
/// [`DecimalVector`]: crate::DecimalVector
/// [`VectorOps`]: crate::VectorOps
#[macro_export]
macro_rules! match_each_dvector {
    ($self:expr, | $vec:ident | $body:block) => {{
        match $self {
            $crate::DecimalVector::D8($vec) => $body,
            $crate::DecimalVector::D16($vec) => $body,
            $crate::DecimalVector::D32($vec) => $body,
            $crate::DecimalVector::D64($vec) => $body,
            $crate::DecimalVector::D128($vec) => $body,
            $crate::DecimalVector::D256($vec) => $body,
        }
    }};
}

/// Matches on all decimal type variants of [`DecimalVectorMut`] and executes the same code
/// for each variant branch.
///
/// This macro eliminates repetitive match statements when implementing mutable operations that need
/// to work uniformly across all decimal type variants (`D8`, `D16`, `D32`, `D64`, `D128`, `D256`).
///
/// # Examples
///
/// ```
/// use vortex_vector::{DecimalVectorMut, DVectorMut, VectorMutOps, match_each_dvector_mut};
///
/// fn reserve_decimal_space(vector: &mut DecimalVectorMut, additional: usize) {
///     match_each_dvector_mut!(vector, |v| { v.reserve(additional) })
/// }
///
/// // Works with `D32` mutable decimal vectors.
/// let mut d32_vec: DecimalVectorMut = DVectorMut::<i32>::from_iter([1, 2].map(Some)).into();
/// reserve_decimal_space(&mut d32_vec, 10);
/// assert!(d32_vec.capacity() >= 12);
///
/// // Works with `D128` mutable decimal vectors.
/// let mut d128_vec: DecimalVectorMut = DVectorMut::<i128>::from_iter([100].map(Some)).into();
/// reserve_decimal_space(&mut d128_vec, 5);
/// assert!(d128_vec.capacity() >= 6);
/// ```
///
/// Note: The `reserve` method is already provided by the [`VectorMutOps`] trait implementation.
///
/// [`DecimalVectorMut`]: crate::DecimalVectorMut
/// [`VectorMutOps`]: crate::VectorMutOps
#[macro_export]
macro_rules! match_each_dvector_mut {
    ($self:expr, | $vec:ident | $body:block) => {{
        match $self {
            $crate::DecimalVectorMut::D8($vec) => $body,
            $crate::DecimalVectorMut::D16($vec) => $body,
            $crate::DecimalVectorMut::D32($vec) => $body,
            $crate::DecimalVectorMut::D64($vec) => $body,
            $crate::DecimalVectorMut::D128($vec) => $body,
            $crate::DecimalVectorMut::D256($vec) => $body,
        }
    }};
}
