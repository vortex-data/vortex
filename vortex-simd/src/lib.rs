//! Runtime CPU feature detection and SIMD kernels for Vortex integer compute.
//!
//! Two layers:
//!
//! 1. [`cpu`] — detect the SIMD [`Tier`](cpu::Tier) once and query it cheaply.
//! 2. [`kernels`] — a single registry of `fn` pointers. Call
//!    [`kernels::kernels`] (re-exported here as [`kernels`]) once per hot
//!    function, then dispatch a kernel via a field:
//!    `(kernels().i32_add)(a, b, out)`. Each slot is either the best
//!    specialized kernel for the active tier or the scalar fallback.
//!
//! # Adding kernels at scale
//!
//! For 1000s of kernels × 5 tiers, write the source once where possible.
//! Element-wise ops (add, sub, mul, ordered compare with byte output) live
//! in [`kernels::generic`] as one `#[inline]` function; the per-tier
//! wrappers in [`arch`] are one-liners that LLVM autovectorizes against
//! the wrapper's `#[target_feature]`. Hand-tuned intrinsics are reserved
//! for kernels with mask packing, lane permutations, or fastlanes-style
//! bit packing where the autovectorizer cannot see the right pattern.
//!
//! Each new tier table is a single static; adding a kernel is one field on
//! [`kernels::Kernels`] plus one line per tier table.

pub mod arch;
pub mod cpu;
pub mod kernels;

pub use cpu::{Tier, has_avx2, has_avx512, has_neon, tier};
pub use kernels::{Kernels, kernels};
