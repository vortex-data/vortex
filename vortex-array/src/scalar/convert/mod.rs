// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`From`] and [`TryFrom`] implementations to and from [`Scalar`]s and [`ScalarValue`]s.
//!
//! This module is broken up into `from_scalar` and `into_scalar` submodules, with the exception of
//! the [`primitive`] and [`decimal`] conversion implementations because they involve a lot of
//! macros.
//!
//! [`Scalar`]: crate::scalar::Scalar
//! [`ScalarValue`]: crate::scalar::ScalarValue

// TODO(connor): Do we want to have stubs for `TryFrom<Scalar>` that just call into
// `TryFrom<&Scalar>`? Sometimes it is nice for writing method chains but it can also hide the fact
// that a clone is happening.

mod decimal;
mod from_scalar;
mod into_scalar;
mod primitive;
