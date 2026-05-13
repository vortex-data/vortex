// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex array implementing pco's IntMult mode: a single unsigned latent
//! stream is decomposed into two ordered latents `primary` and `secondary`
//! related by a fixed `base`. The decoded value is
//! `base.wrapping_mul(primary) + secondary` in the latent type `L`.
//!
//! See [`IntMultArray`] for the public type.

pub use array::*;

mod array;

/// Prost-encoded metadata for an [`IntMultArray`].
///
/// The `base` is stored as `u64` for every supported latent width; the
/// array's encoded child dtype tells the reader which width to interpret it
/// as.
#[derive(Clone, prost::Message)]
pub struct IntMultMetadata {
    /// The multiplier used to decompose each latent value into
    /// `(primary, secondary)`.
    ///
    /// Must be `>= 2` and must fit into the latent type `L`.
    #[prost(uint64, tag = "1")]
    pub base: u64,
}
