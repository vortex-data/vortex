// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::all, clippy::nursery, clippy::absolute_paths)]

#[cfg(feature = "dtype")]
#[rustfmt::skip]
#[path = "./generated/vortex.dtype.rs"]
pub mod dtype;

#[cfg(feature = "scalar")]
#[rustfmt::skip]
#[path = "./generated/vortex.scalar.rs"]
pub mod scalar;

#[cfg(feature = "expr")]
#[rustfmt::skip]
#[path = "./generated/vortex.expr.rs"]
pub mod expr;
