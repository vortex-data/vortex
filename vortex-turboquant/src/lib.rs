// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant vector quantization extension type for Vortex.
//!
//! Implements a Stage 1 TurboQuant encoding ([arXiv:2504.19874], [RFC 0033]) for lossy compression
//! of high-dimensional vector data. The extension operates on
//! [`Vector`](vortex_tensor::vector::Vector) extension arrays, encoding their `FixedSizeList`
//! storage into quantized codes after a structured orthogonal surrogate transform.
//!
//! [arXiv:2504.19874]: https://arxiv.org/abs/2504.19874
//! [RFC 0033]: https://vortex-data.github.io/rfcs/rfc/0033.html
//!
//! # Overview
//!
//! TurboQuant minimizes mean-squared reconstruction error (1-8 bits per coordinate)
//! using MSE-optimal scalar quantization on coordinates of a transformed unit vector.
//!
//! The [`TQEncode`] scalar function first computes and stores the original L2 norm for each vector
//! row, then normalizes each valid nonzero row internally before SORF transform and scalar
//! quantization. The [`TQDecode`] scalar function dequantizes through deterministic centroids,
//! applies the inverse SORF transform, truncates back to the original dimension, and re-applies the
//! stored norm.
//!
//! The encoded storage is a row-aligned extension tree:
//!
//! ```text
//! Extension<TurboQuant>(
//!     Struct {
//!         norms: Primitive<element_ptype, vector_validity>,
//!         codes: FixedSizeList<Primitive<u8>, padded_dim, vector_validity>,
//!     }
//! )
//! ```
//!
//! Stored norms are authoritative for future TurboQuant-aware scalar functions. Decoded quantized
//! directions are not guaranteed to have unit norm after scalar quantization and inverse transform.
//!
//! # Source map
//!
//! Implementation details are documented next to the code that owns them:
//!
//! - `vector/storage.rs`: physical storage shape, full-length child arrays, and field-level
//!   validity for null vectors.
//! - `vector/normalize.rs`: TurboQuant-local normalization and how it differs from the tensor
//!   crate's null-row zeroing helper.
//! - `vector/quantize.rs`: SORF transform, centroid lookup, and why invalid rows are skipped rather
//!   than quantized.
//! - `centroids.rs`: deterministic Max-Lloyd centroid computation and process-local caching.
//! - `sorf/`: the Walsh-Hadamard-based structured transform and the stable SplitMix64 sign stream.
//!
//! The current encoding is intentionally MSE-only. It does not yet implement the paper's QJL
//! residual correction for unbiased inner-product estimation, and it still uses internal
//! power-of-2 padding rather than the block decomposition proposed in RFC 0033.

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

// TODO(connor): We need to somehow make sure that callers call `vortex_tensor::initialize` first.
/// Register the TurboQuant extension type with a Vortex session.
pub fn initialize(session: &vortex_session::VortexSession) {
    use vortex_array::dtype::session::DTypeSessionExt;
    use vortex_array::scalar_fn::session::ScalarFnSessionExt;
    session.dtypes().register(TurboQuant);

    session.scalar_fns().register(TQEncode);
    session.scalar_fns().register(TQDecode);
}

#[cfg(test)]
mod tests;
