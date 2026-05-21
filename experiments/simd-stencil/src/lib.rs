// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! # Copy-and-patch SIMD stencils for columnar decode
//!
//! A prototype exploring how to decode *stacked* Vortex encodings
//! (`delta(bitpacking)`, `alp(delta(ffor(bitpacking)))`,
//! `rle(alp(delta(ffor(bitpacking))))`) using four composition strategies that
//! all share the *same* pre-compiled SIMD kernels ("stencils"):
//!
//! 1. [`strategies::materialized`] — decode each encoding layer into a full-column
//!    buffer before the next layer reads it. This mirrors how Vortex's
//!    array-by-array `execute` path materialises a `PrimitiveArray` per layer.
//! 2. [`strategies::fused`] — a tiled, L1-resident pipeline that selects
//!    pre-compiled stencil functions and patches in runtime constants
//!    (bit-width, FoR reference, ALP scale). This is "copy-and-patch" with the
//!    stencils kept as ordinary function pointers.
//! 3. [`patched`] — true copy-and-patch: the constant-bearing stencil is emitted
//!    as machine code at run time by copying a template into an executable page
//!    and patching the constant in as an immediate. Build cost ~= a `memcpy`.
//! 4. [`strategies::aot`] — the ahead-of-time upper bound: a fully-inlined,
//!    const-generic pipeline monomorphised for the exact (stack, bit-width)
//!    combination, dispatched through a `match` over every width.
//!
//! The point of the prototype is to isolate the *composition strategy* as the
//! only variable, so the kernels (`fastlanes` bit-unpack / undelta / untranspose,
//! plus the ALP scale and RLE expand stencils) are identical across all four.

pub mod encode;
pub mod kernels;
pub mod patched;
pub mod strategies;
pub mod vortex_baseline;

#[cfg(test)]
mod tests;

/// FastLanes tile width: every kernel operates on 1024-element tiles.
pub const TILE: usize = 1024;

/// The encoding stacks the prototype decodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stack {
    /// `delta(bitpacking)` over `u32`.
    DeltaBitpack,
    /// `alp(delta(ffor(bitpacking)))` over `f64`.
    AlpDeltaForBitpack,
    /// `rle(alp(delta(ffor(bitpacking))))`: f64 run values plus delta-bitpacked run ends.
    RleAlpDeltaForBitpack,
}

/// The decode composition strategies under comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Decode each layer into a full-column buffer (Vortex-style).
    Materialized,
    /// Tiled L1-resident pipeline using pre-compiled stencil functions.
    Fused,
    /// Tiled pipeline using a run-time-emitted, immediate-patched machine-code stencil.
    Patched,
    /// Fully-inlined const-generic pipeline (ahead-of-time upper bound).
    Aot,
}
