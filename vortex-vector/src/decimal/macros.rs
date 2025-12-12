// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Helper macros for working with the different variants of [`DecimalVector`], [`DecimalScalar`],
//! and [`DecimalVectorMut`].
//!
//! [`DecimalVector`]: super::DecimalVector
//! [`DecimalScalar`]: super::DecimalScalar
//! [`DecimalVectorMut`]: super::DecimalVectorMut

/// Matches on all decimal type variants of [`DecimalVector`] and executes the same code for
/// each variant branch.
///
/// This macro eliminates repetitive match statements when implementing operations that need to work
/// uniformly across all decimal type variants (`D8`, `D16`, `D32`, `D64`, `D128`, `D256`).
///
/// [`DecimalVector`]: super::DecimalVector
#[macro_export]
macro_rules! match_each_dvector {
    ($self:expr, | $vec:ident | $body:block) => {{
        match $self {
            $crate::decimal::DecimalVector::D8($vec) => $body,
            $crate::decimal::DecimalVector::D16($vec) => $body,
            $crate::decimal::DecimalVector::D32($vec) => $body,
            $crate::decimal::DecimalVector::D64($vec) => $body,
            $crate::decimal::DecimalVector::D128($vec) => $body,
            $crate::decimal::DecimalVector::D256($vec) => $body,
        }
    }};
}

/// Matches on all decimal type variants of [`DecimalVectorMut`] and executes the same code
/// for each variant branch.
///
/// This macro eliminates repetitive match statements when implementing mutable operations that need
/// to work uniformly across all decimal type variants (`D8`, `D16`, `D32`, `D64`, `D128`, `D256`).
///
/// [`DecimalVectorMut`]: super::DecimalVectorMut
#[macro_export]
macro_rules! match_each_dvector_mut {
    ($self:expr, | $vec:ident | $body:block) => {{
        match $self {
            $crate::decimal::DecimalVectorMut::D8($vec) => $body,
            $crate::decimal::DecimalVectorMut::D16($vec) => $body,
            $crate::decimal::DecimalVectorMut::D32($vec) => $body,
            $crate::decimal::DecimalVectorMut::D64($vec) => $body,
            $crate::decimal::DecimalVectorMut::D128($vec) => $body,
            $crate::decimal::DecimalVectorMut::D256($vec) => $body,
        }
    }};
}

/// Matches on all decimal type variants of [`DecimalScalar`] and executes the same code for
/// each variant branch.
///
/// This macro eliminates repetitive match statements when implementing operations that need to work
/// uniformly across all decimal type variants (`D8`, `D16`, `D32`, `D64`, `D128`, `D256`).
///
/// Works with both owned `DecimalScalar` and `&DecimalScalar` (the bound variable will be
/// `DScalar<D>` or `&DScalar<D>` respectively due to Rust's match ergonomics).
///
/// [`DecimalScalar`]: super::DecimalScalar
#[macro_export]
macro_rules! match_each_dscalar {
    ($self:expr, | $scalar:ident | $body:block) => {{
        match $self {
            $crate::decimal::DecimalScalar::D8($scalar) => $body,
            $crate::decimal::DecimalScalar::D16($scalar) => $body,
            $crate::decimal::DecimalScalar::D32($scalar) => $body,
            $crate::decimal::DecimalScalar::D64($scalar) => $body,
            $crate::decimal::DecimalScalar::D128($scalar) => $body,
            $crate::decimal::DecimalScalar::D256($scalar) => $body,
        }
    }};
}
