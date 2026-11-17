// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Test extension types for exercising the [`ExtVTable`](crate::dtype::extension::ExtVTable) contract.

mod divisible_int;
pub(crate) use divisible_int::DivisibleInt;
pub(crate) use divisible_int::Divisor;

mod opaque;
pub(crate) use opaque::Opaque;
