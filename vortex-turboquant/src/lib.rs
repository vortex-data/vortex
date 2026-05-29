// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant vector quantization extension type for Vortex.
//!
//! Implements the TurboQuant encoding ([arXiv:2504.19874]) for lossy compression (quantization) of
//! high-dimensional vector data as a lossy extension type. TurboQuant converts to and from
//! [`Vector`](vortex_tensor::vector::Vector) extension arrays, encoding their `FixedSizeList`
//! storage into quantized codes after a random orthogonal transform (via a Structured Orthogonal
//! Random Features matrix, see the `sorf` module for more information).
//!
//! [arXiv:2504.19874]: https://arxiv.org/abs/2504.19874
//!
//! # Overview
//!
//! TurboQuant minimizes mean-squared reconstruction error (1-8 bits per coordinate) using
//! MSE-optimal scalar quantization on coordinates of a transformed unit vector.
//!
//! Each input vector of `dimensions` coordinates is split into a fixed sequence of contiguous
//! power-of-two-sized slices called **blocks**, whose widths are chosen by the user through
//! [`TurboQuantConfig`] and stored verbatim in [`TurboQuantMetadata::block_sizes`]. The TurboQuant
//! algorithm runs on every block independently: each block has its own stored L2 norm, its own SORF
//! matrix seeded by a distinct derived seed, and its own scalar-quantization centroid table sized
//! to that block's width. Block `i` covers input coordinates
//! `[offset_i .. offset_i + block_sizes[i])` with `offset_i = sum(block_sizes[..i])`; a block
//! extending past `dimensions` is zero-padded on encode, and the reconstructed coordinates past
//! `dimensions` are dropped on decode.
//!
//! The encoded storage is a row-aligned extension tree of one outer struct holding one inner struct
//! per block:
//!
//! ```text
//! Extension<TurboQuant>(
//!     Struct {
//!         block_0: Struct {
//!             norms: Primitive<element_ptype, vector_validity>,
//!             codes: FixedSizeList<Primitive<u8>, block_sizes[0], vector_validity>,
//!         },
//!         ...
//!         block_{N-1}: Struct { norms: ..., codes: FixedSizeList<u8, block_sizes[N-1], ...> },
//!     }
//! )
//! ```
//!
//! IMPORTANT NOTE: Stored norms are authoritative for future TurboQuant-aware scalar functions.
//! Decoded quantized directions are not guaranteed to have unit norm after scalar quantization and
//! inverse transform.
//!
//! # Limitations
//!
//! The current encoding is intentionally MSE-only. It does not yet implement the paper's QJL
//! residual correction for unbiased inner-product estimation.
//!
//! # Source map
//!
//! Implementation details are documented next to the code that owns them:
//!
//! - `config.rs`: the operator-facing [`TurboQuantConfig`] and its bit-width and block-list
//!   validation.
//! - `vtable.rs`: the [`TurboQuant`] extension dtype, [`TurboQuantMetadata`] (including the
//!   `block_sizes` list), its proto wire format, and storage-dtype validation.
//! - `scalar_fns/`: the [`TQEncode`] and [`TQDecode`] scalar functions and the metadata wire
//!   format glue.
//! - `vector/storage.rs`: the row-aligned per-block storage layout and the outer-covers-inner
//!   validity coverage rules.
//! - `vector/quantize.rs`: the block-aware encode pipeline (per-block norm, per-block SORF,
//!   scalar quantization).
//! - `centroids.rs`: deterministic Max-Lloyd centroid computation and process-local caching.
//! - `sorf/`: the Walsh-Hadamard-based structured transform, the stable SplitMix64 sign stream,
//!   and the per-block seed derivation that gives each block its own SORF instance.

mod centroids;
mod config;
mod scalar_fns;
mod sorf;
mod vector;
mod vtable;

pub use config::TurboQuantConfig;
pub use scalar_fns::TQDecode;
pub use scalar_fns::TQEncode;
pub use vtable::TurboQuant;
pub use vtable::TurboQuantMetadata;

// TODO(connor): enforce the `vortex_tensor::initialize` ordering at registration time.
/// Register the TurboQuant extension type with a Vortex session.
///
/// Callers must register `vortex_tensor` on the session first so the `Vector` extension type that
/// TurboQuant converts to and from is available.
pub fn initialize(session: &vortex_session::VortexSession) {
    use vortex_array::dtype::session::DTypeSessionExt;
    use vortex_array::scalar_fn::session::ScalarFnSessionExt;

    session.dtypes().register(TurboQuant);

    session.scalar_fns().register(TQEncode);
    session.scalar_fns().register(TQDecode);
}

#[cfg(test)]
mod tests;
