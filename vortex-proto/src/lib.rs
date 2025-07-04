#![allow(clippy::all, clippy::nursery)]

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
