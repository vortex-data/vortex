//! Runtime CPU feature detection and SIMD kernels for Vortex integer compute.
//!
//! Three layers:
//!
//! 1. [`cpu`] — detect the SIMD [`Tier`](cpu::Tier) once and query it cheaply.
//! 2. [`kernels`] — a single registry of `fn` pointers; `(kernels().i32_add)
//!    (a, b, out)` is one indirect call.
//! 3. [`prim`] — typed front-end. `i32::add(a, b, out)` /
//!    `add::<T>(a, b, out)`. Inlines to the kernels layer with no extra
//!    dispatch cost.
//!
//! # Adding kernels at scale
//!
//! Element-wise ops live in [`kernels::generic`] as one `#[inline]` body;
//! each per-tier wrapper in [`arch`] is a one-liner that LLVM autovectorizes
//! against the wrapper's `#[target_feature]`. Hand-tuned intrinsics are
//! reserved for kernels where the autovectorizer cannot see the right
//! pattern (mask packing, lane permutations, fastlanes-style bit packing).
//!
//! Each tier table inherits from [`kernels::SCALAR_TABLE`] via struct-update,
//! so a tier only spells the slots it specializes — every other slot falls
//! back to scalar automatically. Adding kernel #1001 is one field on
//! [`Kernels`] plus one line per tier that has a real specialization.

pub mod arch;
pub mod cpu;
pub mod kernels;
pub mod prim;

pub use cpu::{Tier, has_avx2, has_avx512, has_neon, tier};
pub use kernels::{Kernels, kernels};
pub use prim::Prim;
