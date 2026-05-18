//! Runtime CPU feature detection and SIMD kernels for Vortex integer compute.
//!
//! Two layers:
//!
//! 1. [`cpu`] — detect the SIMD [`Tier`](cpu::Tier) once and query it cheaply.
//! 2. [`ops`] — per-type kernel tables of `fn` pointers. The slot is either
//!    the best specialized kernel available for this CPU, or the scalar
//!    fallback. There is no second-level dispatch at the call site:
//!    `(i32::ops().add)(a, b, out)` is one indirect call.
//!
//! Per-architecture kernels in [`arch`] and the scalar fallback in
//! [`kernels::scalar`] are also `pub` for callers that have already proved a
//! tier and want zero overhead.

pub mod arch;
pub mod cpu;
pub mod kernels;
pub mod ops;

pub use cpu::{Tier, has_avx2, has_avx512, has_neon, tier};
pub use ops::{IntKernels, IntOps};
