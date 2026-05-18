//! Integer compute kernels.
//!
//! Each kernel has a scalar implementation in [`scalar`] and one or more
//! vectorized implementations under [`super::arch`]. Dispatch into the right
//! one is done via the function-pointer tables in [`super::ops`] or via the
//! [`crate::dispatch!`] macro.

pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub use super::arch::x86_64 as x86;

#[cfg(target_arch = "aarch64")]
pub use super::arch::aarch64 as arm;
