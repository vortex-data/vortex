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
