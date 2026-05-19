// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! # vortex-jit-experiment
//!
//! A focused experiment to make the **kernel boundary** in Vortex concrete and
//! to demonstrate what a JIT can do across it.
//!
//! ## What is the kernel boundary?
//!
//! In Vortex today, an operation like `decode bit-packed u32 → compare > k →
//! produce a mask` runs as **two kernels** connected by a temporary buffer:
//!
//! ```text
//!     packed bytes ──► [ unpack kernel ] ──► [u32; 1024] ──► [ compare kernel ] ──► mask
//!                                          ^^^^^^^^^^^^^^^
//!                                          THE BOUNDARY
//! ```
//!
//! Concretely, in `encodings/fastlanes/src/bitpacking/array/unpack_iter.rs`,
//! the boundary is the `[MaybeUninit<T>; 1024]` field on `UnpackedChunks`.
//! `BitPacking::unchecked_unpack` writes 1024 values into that buffer; a
//! downstream consumer reads from it.
//!
//! The boundary has a contract:
//!
//! - **Unpack side promises**: I will fully initialise `dst[0..1024]` after
//!   you give me a packed chunk and a bit-width.
//! - **Compare side promises**: I will read every element of `src[0..1024]`,
//!   emit one result bit per element, and not look outside the slice.
//!
//! Both sides honour this contract by going through memory. That has costs:
//!
//! 1. **Materialisation**: the 4 KiB buffer is written, then immediately read.
//!    On hot loops the buffer lives in L1 but it's still load/store traffic.
//! 2. **Lost invariants**: the compare kernel sees `[u32; 1024]` and knows
//!    nothing about how those values were produced. It cannot use the fact
//!    that, for bit-width B, every value is in `[0, 2^B)` — so it cannot,
//!    e.g., skip the comparison entirely when `k >= 2^B`.
//! 3. **Dispatch overhead**: each kernel is a separate function call (often
//!    through a vtable), so the compiler cannot inline across the boundary.
//!
//! A JIT can erase the boundary at runtime, once it knows the concrete
//! `(encoding, dtype, bit_width, predicate)` tuple.
//!
//! ## What this crate does
//!
//! For a single concrete operation — `unpack u32 bit-packed with width B,
//! compare to threshold k, write a bitmap` — it provides three
//! implementations:
//!
//! - [`composed`]: two kernels with a temp buffer (today's vx model).
//! - [`fused`]: a single hand-written Rust function with no temp buffer (what
//!   a perfect JIT would emit).
//! - [`jit`]: a Cranelift-generated function that does the same as `fused`,
//!   specialised on `B` at runtime.
//!
//! The `boundary-demo` binary runs all three on the same data, checks
//! equivalence, and prints the Cranelift IR so you can see exactly what
//! crosses the boundary in machine code.
//!
//! ## Why not benchmark against `fastlanes`?
//!
//! Production FastLanes uses a transposed 1024-element layout designed to
//! auto-vectorise into AVX-512. Beating it with a JIT for a single bit-width
//! is a separate, much harder project. We use a **simple linear LSB-first
//! packing** so the layout is identical across all three paths and the only
//! variable is whether the boundary exists.

pub mod composed;
pub mod fused;
pub mod jit;
pub mod pack;

/// Number of elements per chunk. Matches FastLanes' `FL_CHUNK_SIZE` so that
/// downstream consumers (bitmaps, validity, mask combiners) align with the
/// rest of vx.
pub const CHUNK_SIZE: usize = 1024;

/// Number of `u64` words needed to hold a 1024-element bitmap.
pub const MASK_WORDS: usize = CHUNK_SIZE / 64;
