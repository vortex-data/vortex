// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vortex array implementing pco's `FloatQuant` mode: an `f64` input stream is
//! decomposed into a `(primary, secondary)` pair of `u64` children that split
//! each value's raw bit pattern at a fixed quantization boundary `k`.
//!
//! For every input `x`, encode produces:
//!
//! ```text
//! bits         = x[i].to_bits()
//! primary[i]   = bits >> k
//! secondary[i] = bits & ((1u64 << k) - 1)
//! ```
//!
//! and decode reconstructs:
//!
//! ```text
//! bits   = (primary[i] << k) | secondary[i]
//! out[i] = f64::from_bits(bits)
//! ```
//!
//! The round-trip is bit-exact for *every* `f64` input including NaN, both
//! infinities, and `+/- 0.0`. When the low `k` mantissa bits of the input
//! carry no signal (e.g. lossy sensor data, ML weights, anything with bounded
//! effective precision), `secondary` is highly predictable — often all zero —
//! and compresses well in downstream entropy-coded layers.
//!
//! See [`FloatQuantArray`] for the public type.

pub use array::*;

mod array;

/// Prost-encoded metadata for a [`FloatQuantArray`].
///
/// `k` is the number of low-order bits taken from each `f64`'s raw bit
/// pattern to form `secondary`. It must satisfy `1 <= k <= 63`. The value is
/// stored as `u32` to match prost's `uint32` convention; the runtime range
/// fits comfortably.
#[derive(Clone, prost::Message)]
pub struct FloatQuantMetadata {
    /// Number of low-order bits assigned to `secondary` (`1..=63`).
    #[prost(uint32, tag = "1")]
    pub k: u32,
}
