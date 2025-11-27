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
/// ```
/// use vortex_vector::Vector;
/// use vortex_vector::bool::BoolVectorMut;
/// use vortex_vector::null::NullVector;
/// use vortex_vector::{VectorOps, VectorMutOps, match_each_vector};
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
#[macro_export]
macro_rules! match_each_vector {
    ($self:expr, | $vec:ident | $body:block) => {{
        match $self {
            $crate::Vector::Null($vec) => $body,
            $crate::Vector::Bool($vec) => $body,
            $crate::Vector::Decimal($vec) => $body,
            $crate::Vector::Primitive($vec) => $body,
            $crate::Vector::String($vec) => $body,
            $crate::Vector::Binary($vec) => $body,
            $crate::Vector::List($vec) => $body,
            $crate::Vector::FixedSizeList($vec) => $body,
            $crate::Vector::Struct($vec) => $body,
        }
    }};
}

/// Matches on all variants of [`VectorMut`] and executes the same code for each variant branch.
///
/// This macro eliminates repetitive match statements when implementing operations that need to work
/// uniformly across all mutable vector type variants.
///
/// # Examples
///
/// ```
/// use vortex_vector::VectorMut;
/// use vortex_vector::bool::BoolVectorMut;
/// use vortex_vector::null::NullVectorMut;
/// use vortex_vector::{VectorMutOps, match_each_vector_mut};
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
#[macro_export]
macro_rules! match_each_vector_mut {
    ($self:expr, | $vec:ident | $body:block) => {{
        match $self {
            $crate::VectorMut::Null($vec) => $body,
            $crate::VectorMut::Bool($vec) => $body,
            $crate::VectorMut::Decimal($vec) => $body,
            $crate::VectorMut::Primitive($vec) => $body,
            $crate::VectorMut::String($vec) => $body,
            $crate::VectorMut::Binary($vec) => $body,
            $crate::VectorMut::List($vec) => $body,
            $crate::VectorMut::FixedSizeList($vec) => $body,
            $crate::VectorMut::Struct($vec) => $body,
        }
    }};
}

/// Internal macro to generate match arms for vector pairs.
#[doc(hidden)]
#[macro_export]
macro_rules! __match_vector_pair_arms {
    (
        $left:expr,
        $right:expr,
        $enum_left:ident,
        $enum_right:ident,
        $a:ident,
        $b:ident,
        $body:expr
    ) => {{
        match ($left, $right) {
            ($crate::$enum_left::Null($a), $crate::$enum_right::Null($b)) => $body,
            ($crate::$enum_left::Bool($a), $crate::$enum_right::Bool($b)) => $body,
            ($crate::$enum_left::Decimal($a), $crate::$enum_right::Decimal($b)) => $body,
            ($crate::$enum_left::Primitive($a), $crate::$enum_right::Primitive($b)) => $body,
            ($crate::$enum_left::String($a), $crate::$enum_right::String($b)) => $body,
            ($crate::$enum_left::Binary($a), $crate::$enum_right::Binary($b)) => $body,
            ($crate::$enum_left::List($a), $crate::$enum_right::List($b)) => $body,
            ($crate::$enum_left::FixedSizeList($a), $crate::$enum_right::FixedSizeList($b)) => {
                $body
            }
            ($crate::$enum_left::Struct($a), $crate::$enum_right::Struct($b)) => $body,
            _ => ::vortex_error::vortex_panic!("Mismatched vector types"),
        }
    }};
}

/// Matches on pairs of vector variants and executes the same code for matching variant pairs.
///
/// This macro eliminates repetitive match statements when implementing operations that need to work
/// with pairs of vectors where the variants must match.
///
/// Specify the types of the left and right vectors (either `Vector` or `VectorMut`) and the macro
/// generates the appropriate match arms.
///
/// The macro binds the matched inner values to identifiers in the closure that can be used in the
/// body expression.
///
/// # Examples
///
/// ```
/// use vortex_vector::{Vector, VectorMut, VectorMutOps, match_vector_pair};
/// use vortex_vector::bool::{BoolVector, BoolVectorMut};
///
/// fn extend_vector(left: &mut VectorMut, right: &Vector) {
///     match_vector_pair!(left, right, |a: VectorMut, b: Vector| {
///         a.extend_from_vector(b);
///     })
/// }
///
/// let mut mut_vec: VectorMut = BoolVectorMut::from_iter([true, false, true]).into();
/// let vec: Vector = BoolVectorMut::from_iter([false, true]).freeze().into();
///
/// extend_vector(&mut mut_vec, &vec);
/// assert_eq!(mut_vec.len(), 5);
/// ```
///
/// Note that the vectors can also be owned:
///
/// ```
/// use vortex_vector::{Vector, VectorMut, VectorMutOps, match_vector_pair};
/// use vortex_vector::bool::{BoolVector, BoolVectorMut};
///
/// fn extend_vector_owned(mut dest: VectorMut, src: Vector) -> VectorMut {
///     match_vector_pair!(&mut dest, src, |a: VectorMut, b: Vector| {
///         a.extend_from_vector(&b);
///         dest
///     })
/// }
///
/// let mut_vec: VectorMut = BoolVectorMut::from_iter([true, false, true]).into();
/// let vec: Vector = BoolVectorMut::from_iter([false, true]).freeze().into();
///
/// let new_bool_mut = extend_vector_owned(mut_vec, vec);
/// assert_eq!(new_bool_mut.len(), 5);
/// ```
#[macro_export] // DO NOT ADD `#[rustfmt::skip]`!!! https://github.com/rust-lang/rust/pull/52234#issuecomment-903419099
macro_rules! match_vector_pair {
    ($left:expr, $right:expr, | $a:ident : Vector, $b:ident : Vector | $body:expr) => {{ $crate::__match_vector_pair_arms!($left, $right, Vector, Vector, $a, $b, $body) }};
    ($left:expr, $right:expr, | $a:ident : Vector, $b:ident : VectorMut | $body:expr) => {{ $crate::__match_vector_pair_arms!($left, $right, Vector, VectorMut, $a, $b, $body) }};
    ($left:expr, $right:expr, | $a:ident : VectorMut, $b:ident : Vector | $body:expr) => {{ $crate::__match_vector_pair_arms!($left, $right, VectorMut, Vector, $a, $b, $body) }};
    ($left:expr, $right:expr, | $a:ident : VectorMut, $b:ident : VectorMut | $body:expr) => {{ $crate::__match_vector_pair_arms!($left, $right, VectorMut, VectorMut, $a, $b, $body) }};
    ($left:expr, $right:expr, | $a:ident : VectorMut, $b:ident : Scalar | $body:expr) => {{ $crate::__match_vector_pair_arms!($left, $right, VectorMut, Scalar, $a, $b, $body) }};
}
