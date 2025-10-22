// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Helper macros for working with the different variants of [`Vector`](crate::Vector) and
//! [`VectorMut`](crate::VectorMut).

/// Matches on all variants of [`Vector`] and executes the same code for each variant branch.
///
/// This macro eliminates repetitive match statements when implementing operations that need to work
/// uniformly across all vector type variants.
///
/// # Examples
///
/// ```ignore
/// use vortex_vector::{Vector, BoolVectorMut, NullVector, VectorOps, VectorMutOps};
///
/// fn get_vector_length(vector: &Vector) -> usize {
///     match_each_vector!(vector, |v| { v.len() })
/// }
///
/// // Works with `Null` vectors.
/// let null_vec: Vector = NullVector::new(5).into();
/// assert_eq!(get_vector_length(&null_vec), 5);
///
/// // Works with `Bool` vectors.
/// let bool_vec: Vector = BoolVectorMut::from_iter([true, false, true].map(Some))
///     .freeze()
///     .into();
/// assert_eq!(get_vector_length(&bool_vec), 3);
/// ```
///
/// Note: The `len` method is already provided by the [`VectorOps`] trait implementation.
///
/// [`Vector`]: crate::Vector
/// [`VectorOps`]: crate::VectorOps
macro_rules! match_each_vector {
    ($self:expr, | $vec:ident | $body:block) => {{
        match $self {
            $crate::Vector::Null(v) => {
                let $vec = v;
                $body
            }
            $crate::Vector::Bool(v) => {
                let $vec = v;
                $body
            }
            $crate::Vector::Primitive(v) => {
                let $vec = v;
                $body
            }
        }
    }};
}

pub(crate) use match_each_vector;

/// Matches on all variants of [`VectorMut`] and executes the same code for each variant branch.
///
/// This macro eliminates repetitive match statements when implementing operations that need to work
/// uniformly across all mutable vector type variants.
///
/// # Examples
///
/// ```ignore
/// use vortex_vector::{VectorMut, BoolVectorMut, NullVectorMut, VectorMutOps};
///
/// fn reserve_space(vector: &mut VectorMut, additional: usize) {
///     match_each_vector_mut!(vector, |v| { v.reserve(additional) })
/// }
///
/// // Works with `Null` mutable vectors.
/// let mut null_vec: VectorMut = NullVectorMut::new(5).into();
/// reserve_space(&mut null_vec, 10);
/// assert!(null_vec.capacity() >= 15);
///
/// // Works with `Bool` mutable vectors.
/// let mut bool_vec: VectorMut = BoolVectorMut::from_iter([true, false].map(Some)).into();
/// reserve_space(&mut bool_vec, 5);
/// assert!(bool_vec.capacity() >= 7);
/// ```
///
/// Note: The `reserve` method is already provided by the [`VectorMutOps`] trait implementation.
///
/// [`VectorMut`]: crate::VectorMut
/// [`VectorMutOps`]: crate::VectorMutOps
macro_rules! match_each_vector_mut {
    ($self:expr, | $vec:ident | $body:block) => {{
        match $self {
            $crate::VectorMut::Null(v) => {
                let $vec = v;
                $body
            }
            $crate::VectorMut::Bool(v) => {
                let $vec = v;
                $body
            }
            $crate::VectorMut::Primitive(v) => {
                let $vec = v;
                $body
            }
        }
    }};
}

pub(crate) use match_each_vector_mut;
