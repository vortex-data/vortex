// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Helper macros for working with the different variants of [`PrimitiveVector`] and
//! [`PrimitiveVectorMut`].
//!
//! [`PrimitiveVector`]: crate::primitive::PrimitiveVector
//! [`PrimitiveVectorMut`]: crate::primitive::PrimitiveVectorMut

/// Matches on all primitive type variants of [`PrimitiveVector`] and executes the same code for
/// each variant branch.
///
/// This macro eliminates repetitive match statements when implementing operations that need to work
/// uniformly across all primitive type variants (`U8`, `U16`, `U32`, `U64`, `I8`, `I16`, `I32`,
/// `I64`, `F16`, `F32`, `F64`).
///
/// # Examples
///
/// ```
/// use vortex_vector::primitive::{PrimitiveVector, PVectorMut};
/// use vortex_vector::{VectorOps, VectorMutOps, match_each_pvector};
///
/// fn get_primitive_len(vector: &PrimitiveVector) -> usize {
///     match_each_pvector!(vector, |v| { v.len() })
/// }
///
/// // Works with `I32` primitive vectors.
/// let i32_vec: PrimitiveVector = PVectorMut::<i32>::from_iter([1, 2, 3].map(Some))
///     .freeze()
///     .into();
/// assert_eq!(get_primitive_len(&i32_vec), 3);
///
/// // Works with `F64` primitive vectors.
/// let f64_vec: PrimitiveVector = PVectorMut::<f64>::from_iter([1.0, 2.5].map(Some))
///     .freeze()
///     .into();
/// assert_eq!(get_primitive_len(&f64_vec), 2);
/// ```
///
/// Note: The `len` method is already provided by the [`VectorOps`] trait implementation.
///
/// [`PrimitiveVector`]: crate::primitive::PrimitiveVector
/// [`VectorOps`]: crate::VectorOps
#[macro_export]
macro_rules! match_each_pvector {
    ($self:expr, | $vec:ident | $body:block) => {{
        match $self {
            $crate::primitive::PrimitiveVector::U8($vec) => $body,
            $crate::primitive::PrimitiveVector::U16($vec) => $body,
            $crate::primitive::PrimitiveVector::U32($vec) => $body,
            $crate::primitive::PrimitiveVector::U64($vec) => $body,
            $crate::primitive::PrimitiveVector::I8($vec) => $body,
            $crate::primitive::PrimitiveVector::I16($vec) => $body,
            $crate::primitive::PrimitiveVector::I32($vec) => $body,
            $crate::primitive::PrimitiveVector::I64($vec) => $body,
            $crate::primitive::PrimitiveVector::F16($vec) => $body,
            $crate::primitive::PrimitiveVector::F32($vec) => $body,
            $crate::primitive::PrimitiveVector::F64($vec) => $body,
        }
    }};
}

/// Matches on all integer type variants of [`PrimitiveVector`] and executes the same code for each
/// of the integer variant branches.
///
/// This macro eliminates repetitive match statements when implementing operations that need to work
/// uniformly across all integer type variants (`U8`, `U16`, `U32`, `U64`, `I8`, `I16`, `I32`,
/// `I64`).
///
/// See [`match_each_pvector`] for similar usage.
///
/// [`PrimitiveVector`]: crate::primitive::PrimitiveVector
///
/// # Panics
///
/// Panics if the vector passed in to the macro is a float vector variant.
#[macro_export]
macro_rules! match_each_integer_pvector {
    ($self:expr, | $vec:ident | $body:block) => {{
        match $self {
            $crate::primitive::PrimitiveVector::U8($vec) => $body,
            $crate::primitive::PrimitiveVector::U16($vec) => $body,
            $crate::primitive::PrimitiveVector::U32($vec) => $body,
            $crate::primitive::PrimitiveVector::U64($vec) => $body,
            $crate::primitive::PrimitiveVector::I8($vec) => $body,
            $crate::primitive::PrimitiveVector::I16($vec) => $body,
            $crate::primitive::PrimitiveVector::I32($vec) => $body,
            $crate::primitive::PrimitiveVector::I64($vec) => $body,
            $crate::primitive::PrimitiveVector::F16(_)
            | $crate::primitive::PrimitiveVector::F32(_)
            | $crate::primitive::PrimitiveVector::F64(_) => {
                ::vortex_error::vortex_panic!(
                    "Tried to match a float vector in an integer match statement"
                )
            }
        }
    }};
}

/// Matches on all unsigned type variants of [`PrimitiveVector`] and executes the same code for each
/// of the unsigned variant branches.
///
/// This macro eliminates repetitive match statements when implementing operations that need to work
/// uniformly across all unsigned type variants (`U8`, `U16`, `U32`, `U64`).
///
/// See [`match_each_pvector`] for similar usage.
///
/// [`PrimitiveVector`]: crate::primitive::PrimitiveVector
///
/// # Panics
///
/// Panics if the vector passed in to the macro is not an unsigned vector variant.
#[macro_export]
macro_rules! match_each_unsigned_pvector {
    ($self:expr, | $vec:ident | $body:block) => {{
        match $self {
            $crate::primitive::PrimitiveVector::U8($vec) => $body,
            $crate::primitive::PrimitiveVector::U16($vec) => $body,
            $crate::primitive::PrimitiveVector::U32($vec) => $body,
            $crate::primitive::PrimitiveVector::U64($vec) => $body,
            $crate::primitive::PrimitiveVector::I8(_)
            | $crate::primitive::PrimitiveVector::I16(_)
            | $crate::primitive::PrimitiveVector::I32(_)
            | $crate::primitive::PrimitiveVector::I64(_)
            | $crate::primitive::PrimitiveVector::F16(_)
            | $crate::primitive::PrimitiveVector::F32(_)
            | $crate::primitive::PrimitiveVector::F64(_) => {
                ::vortex_error::vortex_panic!(
                    "Tried to match a non-unsigned vector in an unsigned match statement"
                )
            }
        }
    }};
}

/// Matches on all primitive type variants of [`PrimitiveVectorMut`] and executes the same code
/// for each variant branch.
///
/// This macro eliminates repetitive match statements when implementing mutable operations that need
/// to work uniformly across all primitive type variants (`U8`, `U16`, `U32`, `U64`, `I8`, `I16`,
/// `I32`, `I64`, `F16`, `F32`, `F64`).
///
/// # Examples
///
/// ```
/// use vortex_vector::primitive::{PrimitiveVectorMut, PVectorMut};
/// use vortex_vector::{VectorMutOps, match_each_pvector_mut};
///
/// fn reserve_primitive_space(vector: &mut PrimitiveVectorMut, additional: usize) {
///     match_each_pvector_mut!(vector, |v| { v.reserve(additional) })
/// }
///
/// // Works with `U8` mutable primitive vectors.
/// let mut u8_vec: PrimitiveVectorMut = PVectorMut::<u8>::from_iter([1, 2].map(Some)).into();
/// reserve_primitive_space(&mut u8_vec, 10);
/// assert!(u8_vec.capacity() >= 12);
///
/// // Works with `I64` mutable primitive vectors.
/// let mut i64_vec: PrimitiveVectorMut = PVectorMut::<i64>::from_iter([100].map(Some)).into();
/// reserve_primitive_space(&mut i64_vec, 5);
/// assert!(i64_vec.capacity() >= 6);
/// ```
///
/// Note: The `reserve` method is already provided by the [`VectorMutOps`] trait implementation.
///
/// [`PrimitiveVectorMut`]: crate::primitive::PrimitiveVectorMut
/// [`VectorMutOps`]: crate::VectorMutOps
#[macro_export]
macro_rules! match_each_pvector_mut {
    ($self:expr, | $vec:ident | $body:block) => {{
        match $self {
            $crate::primitive::PrimitiveVectorMut::U8($vec) => $body,
            $crate::primitive::PrimitiveVectorMut::U16($vec) => $body,
            $crate::primitive::PrimitiveVectorMut::U32($vec) => $body,
            $crate::primitive::PrimitiveVectorMut::U64($vec) => $body,
            $crate::primitive::PrimitiveVectorMut::I8($vec) => $body,
            $crate::primitive::PrimitiveVectorMut::I16($vec) => $body,
            $crate::primitive::PrimitiveVectorMut::I32($vec) => $body,
            $crate::primitive::PrimitiveVectorMut::I64($vec) => $body,
            $crate::primitive::PrimitiveVectorMut::F16($vec) => $body,
            $crate::primitive::PrimitiveVectorMut::F32($vec) => $body,
            $crate::primitive::PrimitiveVectorMut::F64($vec) => $body,
        }
    }};
}

/// Matches on all integer type variants of [`PrimitiveVectorMut`] and executes the same code for
/// each of the integer variant branches.
///
/// This macro eliminates repetitive match statements when implementing operations that need to work
/// uniformly across all integer type variants (`U8`, `U16`, `U32`, `U64`, `I8`, `I16`, `I32`,
/// `I64`).
///
/// See [`match_each_pvector_mut`] for similar usage.
///
/// [`PrimitiveVectorMut`]: crate::primitive::PrimitiveVectorMut
///
/// # Panics
///
/// Panics if the vector passed in to the macro is a float vector variant.
#[macro_export]
macro_rules! match_each_integer_pvector_mut {
    ($self:expr, | $vec:ident | $body:block) => {{
        match $self {
            $crate::primitive::PrimitiveVectorMut::U8($vec) => $body,
            $crate::primitive::PrimitiveVectorMut::U16($vec) => $body,
            $crate::primitive::PrimitiveVectorMut::U32($vec) => $body,
            $crate::primitive::PrimitiveVectorMut::U64($vec) => $body,
            $crate::primitive::PrimitiveVectorMut::I8($vec) => $body,
            $crate::primitive::PrimitiveVectorMut::I16($vec) => $body,
            $crate::primitive::PrimitiveVectorMut::I32($vec) => $body,
            $crate::primitive::PrimitiveVectorMut::I64($vec) => $body,
            $crate::primitive::PrimitiveVectorMut::F16(_)
            | $crate::primitive::PrimitiveVectorMut::F32(_)
            | $crate::primitive::PrimitiveVectorMut::F64(_) => {
                ::vortex_error::vortex_panic!(
                    "Tried to match a mutable float vector in an integer match statement"
                )
            }
        }
    }};
}

/// Matches on all unsigned type variants of [`PrimitiveVectorMut`] and executes the same code for
/// each of the unsigned variant branches.
///
/// This macro eliminates repetitive match statements when implementing operations that need to work
/// uniformly across all unsigned type variants (`U8`, `U16`, `U32`, `U64`).
///
/// See [`match_each_pvector_mut`] for similar usage.
///
/// [`PrimitiveVectorMut`]: crate::primitive::PrimitiveVectorMut
///
/// # Panics
///
/// Panics if the vector passed in to the macro is not an unsigned vector variant.
#[macro_export]
macro_rules! match_each_unsigned_pvector_mut {
    ($self:expr, | $vec:ident | $body:block) => {{
        match $self {
            $crate::primitive::PrimitiveVectorMut::U8($vec) => $body,
            $crate::primitive::PrimitiveVectorMut::U16($vec) => $body,
            $crate::primitive::PrimitiveVectorMut::U32($vec) => $body,
            $crate::primitive::PrimitiveVectorMut::U64($vec) => $body,
            $crate::primitive::PrimitiveVectorMut::I8(_)
            | $crate::primitive::PrimitiveVectorMut::I16(_)
            | $crate::primitive::PrimitiveVectorMut::I32(_)
            | $crate::primitive::PrimitiveVectorMut::I64(_)
            | $crate::primitive::PrimitiveVectorMut::F16(_)
            | $crate::primitive::PrimitiveVectorMut::F32(_)
            | $crate::primitive::PrimitiveVectorMut::F64(_) => {
                ::vortex_error::vortex_panic!(
                    "Tried to match a non-unsigned mutable vector in an unsigned match statement"
                )
            }
        }
    }};
}

/// Matches on pairs of [`crate::primitive::PrimitiveVector`] with the same type and executes the provided code.
#[macro_export]
macro_rules! match_each_pvector_pair {
    (($left:expr, $right:expr), | $l:ident, $r:ident | $body:block, $else:block) => {{
        match ($left, $right) {
            (
                $crate::primitive::PrimitiveVector::U8($l),
                $crate::primitive::PrimitiveVector::U8($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveVector::U16($l),
                $crate::primitive::PrimitiveVector::U16($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveVector::U32($l),
                $crate::primitive::PrimitiveVector::U32($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveVector::U64($l),
                $crate::primitive::PrimitiveVector::U64($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveVector::I8($l),
                $crate::primitive::PrimitiveVector::I8($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveVector::I16($l),
                $crate::primitive::PrimitiveVector::I16($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveVector::I32($l),
                $crate::primitive::PrimitiveVector::I32($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveVector::I64($l),
                $crate::primitive::PrimitiveVector::I64($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveVector::F16($l),
                $crate::primitive::PrimitiveVector::F16($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveVector::F32($l),
                $crate::primitive::PrimitiveVector::F32($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveVector::F64($l),
                $crate::primitive::PrimitiveVector::F64($r),
            ) => $body,
            _ => $else,
        }
    }};
}

/// Matches on pairs of integer [`PrimitiveVector`] with the same type and executes the provided
/// code.
///
/// This macro matches two primitive vectors when they have the same underlying integer type.
/// For type mismatches, the `$else` block is executed.
///
/// [`PrimitiveVector`]: crate::primitive::PrimitiveVector
#[macro_export]
macro_rules! match_each_integer_pvector_pair {
    (($left:expr, $right:expr), | $l:ident, $r:ident | $body:block, $else:block) => {{
        match ($left, $right) {
            (
                $crate::primitive::PrimitiveVector::U8($l),
                $crate::primitive::PrimitiveVector::U8($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveVector::U16($l),
                $crate::primitive::PrimitiveVector::U16($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveVector::U32($l),
                $crate::primitive::PrimitiveVector::U32($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveVector::U64($l),
                $crate::primitive::PrimitiveVector::U64($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveVector::I8($l),
                $crate::primitive::PrimitiveVector::I8($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveVector::I16($l),
                $crate::primitive::PrimitiveVector::I16($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveVector::I32($l),
                $crate::primitive::PrimitiveVector::I32($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveVector::I64($l),
                $crate::primitive::PrimitiveVector::I64($r),
            ) => $body,
            _ => $else,
        }
    }};
}

/// Matches on pairs of float [`PrimitiveVector`] with the same type and executes the provided code.
///
/// This macro matches two primitive vectors when they have the same underlying float type.
/// For type mismatches, the `$else` block is executed.
///
/// [`PrimitiveVector`]: crate::primitive::PrimitiveVector
#[macro_export]
macro_rules! match_each_float_pvector_pair {
    (($left:expr, $right:expr), | $l:ident, $r:ident | $body:block, | $l1:ident, $r2:ident | $else:block) => {{
        match ($left, $right) {
            (
                $crate::primitive::PrimitiveVector::F16($l),
                $crate::primitive::PrimitiveVector::F16($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveVector::F32($l),
                $crate::primitive::PrimitiveVector::F32($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveVector::F64($l),
                $crate::primitive::PrimitiveVector::F64($r),
            ) => $body,
            ($l1, $r2) => $else,
        }
    }};
}

/// Matches on all primitive type variants of [`PrimitiveScalar`] and executes the same code for
/// each variant branch.
///
/// This macro eliminates repetitive match statements when implementing operations that need to work
/// uniformly across all primitive type variants (`U8`, `U16`, `U32`, `U64`, `I8`, `I16`, `I32`,
/// `I64`, `F16`, `F32`, `F64`).
///
/// Works with both owned `PrimitiveScalar` and `&PrimitiveScalar` (the bound variable will be
/// `PScalar<T>` or `&PScalar<T>` respectively due to Rust's match ergonomics).
///
/// [`PrimitiveScalar`]: crate::primitive::PrimitiveScalar
#[macro_export]
macro_rules! match_each_pscalar {
    ($self:expr, | $scalar:ident | $body:block) => {{
        match $self {
            $crate::primitive::PrimitiveScalar::U8($scalar) => $body,
            $crate::primitive::PrimitiveScalar::U16($scalar) => $body,
            $crate::primitive::PrimitiveScalar::U32($scalar) => $body,
            $crate::primitive::PrimitiveScalar::U64($scalar) => $body,
            $crate::primitive::PrimitiveScalar::I8($scalar) => $body,
            $crate::primitive::PrimitiveScalar::I16($scalar) => $body,
            $crate::primitive::PrimitiveScalar::I32($scalar) => $body,
            $crate::primitive::PrimitiveScalar::I64($scalar) => $body,
            $crate::primitive::PrimitiveScalar::F16($scalar) => $body,
            $crate::primitive::PrimitiveScalar::F32($scalar) => $body,
            $crate::primitive::PrimitiveScalar::F64($scalar) => $body,
        }
    }};
}

/// Matches on pairs of [`crate::primitive::PrimitiveScalar`] with the same type and executes the provided code.
///
/// This macro matches two primitive scalars when they have the same underlying type.
/// For type mismatches, the `$else` block is executed.
///
/// [`PrimitiveScalar`]: crate::primitive::PrimitiveScalar
#[macro_export]
macro_rules! match_each_pscalar_pair {
    (($left:expr, $right:expr), | $l:ident, $r:ident | $body:block, $else:block) => {{
        match ($left, $right) {
            (
                $crate::primitive::PrimitiveScalar::U8($l),
                $crate::primitive::PrimitiveScalar::U8($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveScalar::U16($l),
                $crate::primitive::PrimitiveScalar::U16($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveScalar::U32($l),
                $crate::primitive::PrimitiveScalar::U32($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveScalar::U64($l),
                $crate::primitive::PrimitiveScalar::U64($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveScalar::I8($l),
                $crate::primitive::PrimitiveScalar::I8($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveScalar::I16($l),
                $crate::primitive::PrimitiveScalar::I16($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveScalar::I32($l),
                $crate::primitive::PrimitiveScalar::I32($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveScalar::I64($l),
                $crate::primitive::PrimitiveScalar::I64($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveScalar::F16($l),
                $crate::primitive::PrimitiveScalar::F16($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveScalar::F32($l),
                $crate::primitive::PrimitiveScalar::F32($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveScalar::F64($l),
                $crate::primitive::PrimitiveScalar::F64($r),
            ) => $body,
            _ => $else,
        }
    }};
}

/// Matches on pairs of integer [`PrimitiveScalar`] with the same type and executes the provided
/// code.
///
/// This macro matches two primitive scalars when they have the same underlying integer type.
/// For type mismatches, the `$else` block is executed.
///
/// [`PrimitiveScalar`]: crate::primitive::PrimitiveScalar
#[macro_export]
macro_rules! match_each_integer_pscalar_pair {
    (($left:expr, $right:expr), | $l:ident, $r:ident | $body:block, $else:block) => {{
        match ($left, $right) {
            (
                $crate::primitive::PrimitiveScalar::U8($l),
                $crate::primitive::PrimitiveScalar::U8($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveScalar::U16($l),
                $crate::primitive::PrimitiveScalar::U16($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveScalar::U32($l),
                $crate::primitive::PrimitiveScalar::U32($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveScalar::U64($l),
                $crate::primitive::PrimitiveScalar::U64($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveScalar::I8($l),
                $crate::primitive::PrimitiveScalar::I8($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveScalar::I16($l),
                $crate::primitive::PrimitiveScalar::I16($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveScalar::I32($l),
                $crate::primitive::PrimitiveScalar::I32($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveScalar::I64($l),
                $crate::primitive::PrimitiveScalar::I64($r),
            ) => $body,
            _ => $else,
        }
    }};
}

/// Matches on pairs of float [`PrimitiveScalar`] with the same type and executes the provided code.
///
/// This macro matches two primitive scalars when they have the same underlying float type.
/// For type mismatches, the `$else` block is executed.
///
/// [`PrimitiveScalar`]: crate::primitive::PrimitiveScalar
#[macro_export]
macro_rules! match_each_float_pscalar_pair {
    (($left:expr, $right:expr), | $l:ident, $r:ident | $body:block, $else:block) => {{
        match ($left, $right) {
            (
                $crate::primitive::PrimitiveScalar::F16($l),
                $crate::primitive::PrimitiveScalar::F16($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveScalar::F32($l),
                $crate::primitive::PrimitiveScalar::F32($r),
            ) => $body,
            (
                $crate::primitive::PrimitiveScalar::F64($l),
                $crate::primitive::PrimitiveScalar::F64($r),
            ) => $body,
            _ => $else,
        }
    }};
}
