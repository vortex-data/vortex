//! Per-architecture vectorized kernels.
//!
//! These modules are only compiled on the architecture they target. They are
//! re-exported from [`crate::kernels`] for ergonomic access from dispatch
//! code.

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;
