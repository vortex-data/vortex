// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Helper macros for working with the different variants of [`PrimitiveVector`] and
//! [`PrimitiveVectorMut`].
//!
//! All macros are exported at the crate level with `#[macro_export]`.
//!
//! [`PrimitiveVector`]: crate::PrimitiveVector
//! [`PrimitiveVectorMut`]: crate::PrimitiveVectorMut

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
/// use vortex_vector::{PrimitiveVector, PVectorMut, VectorOps, VectorMutOps, match_each_pvector};
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
/// [`PrimitiveVector`]: crate::PrimitiveVector
/// [`VectorOps`]: crate::VectorOps
#[macro_export]
macro_rules! match_each_pvector {
    ($self:expr, | $vec:ident | $body:block) => {{
        match $self {
            $crate::PrimitiveVector::U8(v) => {
                let $vec = v;
                $body
            }
            $crate::PrimitiveVector::U16(v) => {
                let $vec = v;
                $body
            }
            $crate::PrimitiveVector::U32(v) => {
                let $vec = v;
                $body
            }
            $crate::PrimitiveVector::U64(v) => {
                let $vec = v;
                $body
            }
            $crate::PrimitiveVector::I8(v) => {
                let $vec = v;
                $body
            }
            $crate::PrimitiveVector::I16(v) => {
                let $vec = v;
                $body
            }
            $crate::PrimitiveVector::I32(v) => {
                let $vec = v;
                $body
            }
            $crate::PrimitiveVector::I64(v) => {
                let $vec = v;
                $body
            }
            $crate::PrimitiveVector::F16(v) => {
                let $vec = v;
                $body
            }
            $crate::PrimitiveVector::F32(v) => {
                let $vec = v;
                $body
            }
            $crate::PrimitiveVector::F64(v) => {
                let $vec = v;
                $body
            }
        }
    }};
}

// TODO(connor): Make this a proper Rust test after we replace `BooleanBuffer` with `BitBuffer`
// in `MaskValues`.
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
/// use vortex_vector::{PrimitiveVectorMut, PVectorMut, VectorMutOps, match_each_pvector_mut};
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
/// [`PrimitiveVectorMut`]: crate::PrimitiveVectorMut
/// [`VectorMutOps`]: crate::VectorMutOps
#[macro_export]
macro_rules! match_each_pvector_mut {
    ($self:expr, | $vec:ident | $body:block) => {{
        match $self {
            $crate::PrimitiveVectorMut::U8(v) => {
                let $vec = v;
                $body
            }
            $crate::PrimitiveVectorMut::U16(v) => {
                let $vec = v;
                $body
            }
            $crate::PrimitiveVectorMut::U32(v) => {
                let $vec = v;
                $body
            }
            $crate::PrimitiveVectorMut::U64(v) => {
                let $vec = v;
                $body
            }
            $crate::PrimitiveVectorMut::I8(v) => {
                let $vec = v;
                $body
            }
            $crate::PrimitiveVectorMut::I16(v) => {
                let $vec = v;
                $body
            }
            $crate::PrimitiveVectorMut::I32(v) => {
                let $vec = v;
                $body
            }
            $crate::PrimitiveVectorMut::I64(v) => {
                let $vec = v;
                $body
            }
            $crate::PrimitiveVectorMut::F16(v) => {
                let $vec = v;
                $body
            }
            $crate::PrimitiveVectorMut::F32(v) => {
                let $vec = v;
                $body
            }
            $crate::PrimitiveVectorMut::F64(v) => {
                let $vec = v;
                $body
            }
        }
    }};
}
