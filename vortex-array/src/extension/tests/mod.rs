// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Test extension types for exercising the [`ExtVTable`](crate::dtype::extension::ExtVTable) contract.

mod claims_eq;
pub(crate) use claims_eq::ClaimsEq;

mod decimal_like;
pub(crate) use decimal_like::DecimalLike;

mod divisible_int;
pub(crate) use divisible_int::DivisibleInt;
pub(crate) use divisible_int::Divisor;

mod opaque;
pub(crate) use opaque::Opaque;

mod refined;
pub(crate) use refined::Refined;
