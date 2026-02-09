// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`From`] and [`TryFrom`] implementations to and from [`Scalar`]s and [`ScalarValue`]s.
//!
//! This module is broken up into `from_scalar` and `into_scalar` submodules, with the exception of
//! the [`primitive`] and [`decimal`] conversion implementations because they involve a lot of
//! macros.
//!
//! [`Scalar`]: crate::Scalar
//! [`ScalarValue`]: crate::ScalarValue

mod decimal;
mod from_scalar;
mod into_scalar;
mod primitive;
